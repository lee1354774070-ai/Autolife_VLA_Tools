"""Small authenticated JSON/JPEG protocol for remote LingBot chunk inference."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


MAX_REQUEST_BYTES = 32 * 1024 * 1024


class RemoteInferenceError(RuntimeError):
    """A transport or server-side remote inference failure."""


def encode_jpeg(image_bgr: np.ndarray, *, size: int = 256, quality: int = 90) -> str:
    """Resize one BGR camera frame and return base64 JPEG text."""

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected a BGR image, got {image.shape}.")
    image = cv2.resize(image[..., :3], (size, size), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("OpenCV could not JPEG-encode a camera frame.")
    return base64.b64encode(encoded).decode("ascii")


def decode_jpeg(value: str) -> np.ndarray:
    """Decode base64 JPEG into normalized RGB CHW float32."""

    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 camera frame.") from exc
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode a camera JPEG.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0


class LingBotRemoteClient:
    """Blocking HTTP client used only at action-chunk boundaries."""

    def __init__(self, server_url: str, token: str, timeout_sec: float) -> None:
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.timeout_sec = timeout_sec

    def _request(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.server_url + endpoint,
            data=data,
            method="GET" if data is None else "POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                message = body
            raise RemoteInferenceError(f"Remote HTTP {exc.code}: {message}") from exc
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RemoteInferenceError(f"Remote inference transport failed: {exc}") from exc
        if not isinstance(result, dict):
            raise RemoteInferenceError("Remote server returned a non-object response.")
        if result.get("error"):
            raise RemoteInferenceError(str(result["error"]))
        return result

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def start(self, task: str, images: dict[str, str]) -> dict[str, Any]:
        return self._request("/start", {"task": task, "images": images})

    def infer(self, session_id: str, keyframes: list[dict[str, str]]) -> dict[str, Any]:
        return self._request(
            "/infer",
            {"session_id": session_id, "keyframes": keyframes},
        )

    def close(self, session_id: str) -> None:
        try:
            self._request("/close", {"session_id": session_id})
        except RemoteInferenceError:
            pass


def validate_action_response(response: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16 or len(actions) == 0:
        raise RemoteInferenceError(f"Remote action chunk has invalid shape {actions.shape}.")
    if not np.isfinite(actions).all():
        raise RemoteInferenceError("Remote action chunk contains non-finite values.")
    return actions


def validate_policy_metadata(
    response: dict[str, Any],
) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """Validate authenticated checkpoint metadata returned by the inference server."""

    camera_keys = response.get("camera_keys")
    action_channels = response.get("action_channels")
    if (
        not isinstance(camera_keys, list)
        or len(camera_keys) != 3
        or not all(isinstance(key, str) and key for key in camera_keys)
    ):
        raise RemoteInferenceError(f"Remote server returned invalid camera keys: {camera_keys!r}.")
    if (
        not isinstance(action_channels, list)
        or len(action_channels) != 16
        or not all(isinstance(channel, int) for channel in action_channels)
    ):
        raise RemoteInferenceError(
            f"Remote server returned invalid action channels: {action_channels!r}."
        )

    try:
        lower = np.asarray(response.get("action_min"), dtype=np.float32).reshape(-1)
        upper = np.asarray(response.get("action_max"), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RemoteInferenceError("Remote server returned non-numeric action bounds.") from exc
    if (
        lower.shape != (16,)
        or upper.shape != (16,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.all(lower <= upper)
    ):
        raise RemoteInferenceError(
            f"Remote server returned invalid action bounds: min={lower.shape}, max={upper.shape}."
        )
    return camera_keys, action_channels, lower, upper
