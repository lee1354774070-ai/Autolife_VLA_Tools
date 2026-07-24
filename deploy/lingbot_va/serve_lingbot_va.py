#!/usr/bin/env python3
"""Serve one stateful LingBot-VA policy over an authenticated localhost HTTP API."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR.parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from lingbot_va.lingbot_remote_protocol import MAX_REQUEST_BYTES, decode_jpeg
from lingbot_va.lingbot_va_policy import (
    action_tensor,
    load_checkpoint_action_bounds,
    load_lingbot_policy,
    unwrap_lingbot_policy,
)


def _read_token(value: str | None, token_file: Path | None) -> str:
    if value:
        return value.strip()
    if token_file is None:
        raise ValueError("Set --token, --token-file, or LINGBOT_REMOTE_TOKEN.")
    try:
        token = token_file.expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Cannot read remote token file {token_file}: {exc}") from exc
    if not token:
        raise ValueError(f"Remote token file is empty: {token_file}")
    return token


class LingBotRemoteEngine:
    """Own one policy and its autoregressive cache for one active robot session."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.lock = threading.RLock()
        print("Loading LingBot-VA policy...", flush=True)
        self.policy, self.preprocessor, self.postprocessor = load_lingbot_policy(
            args.model_dir,
            device=args.device,
            base_model=args.base_model,
            wan_model=args.wan_model,
            text_encoder_device=args.text_encoder_device,
            attn_mode=args.attn_mode,
            action_inference_steps=args.action_inference_steps,
            video_inference_steps=args.video_inference_steps,
            guidance_scale=args.guidance_scale,
            offline=args.offline,
        )
        self.base_policy = unwrap_lingbot_policy(self.policy)
        self.camera_keys = tuple(self.base_policy.config.obs_cam_keys)
        if len(self.camera_keys) != 3:
            raise ValueError(f"Remote deployment expects three camera keys, got {self.camera_keys}.")
        self.action_lower_bound, self.action_upper_bound = load_checkpoint_action_bounds(
            args.model_dir
        )
        self.session_id: str | None = None
        self.task = ""
        self.last_chunk_size = 0
        print(
            f"LingBot-VA ready: cameras={self.camera_keys}, "
            f"channels={self.base_policy.config.used_action_channel_ids}",
            flush=True,
        )

    def _reset(self) -> None:
        self.base_policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if hasattr(processor, "reset"):
                processor.reset()

    def _observation(self, images: dict[str, str], task: str) -> dict[str, Any]:
        if set(images) != set(self.camera_keys):
            raise ValueError(
                f"Camera keys must be {sorted(self.camera_keys)}, got {sorted(images)}."
            )
        observation: dict[str, Any] = {"task": task}
        for key in self.camera_keys:
            observation[key] = torch.from_numpy(decode_jpeg(images[key]))
        return self.preprocessor(observation)

    def _physical_actions(self, normalized_actions: torch.Tensor) -> list[list[float]]:
        processed = action_tensor(self.postprocessor(normalized_actions))
        if processed.ndim == 3 and processed.shape[0] == 1:
            processed = processed.squeeze(0)
        if processed.ndim != 2 or processed.shape[1] != 16:
            raise ValueError(f"Postprocessor returned action shape {tuple(processed.shape)}.")
        return processed.detach().to(torch.float32).cpu().tolist()

    def _response(self, actions: torch.Tensor, elapsed_sec: float) -> dict[str, Any]:
        physical = self._physical_actions(actions)
        self.last_chunk_size = len(physical)
        return {
            "session_id": self.session_id,
            "actions": physical,
            "inference_ms": elapsed_sec * 1000.0,
            "chunk_size": self.last_chunk_size,
        }

    def start(self, task: str, images: dict[str, str]) -> dict[str, Any]:
        task = task.strip()
        if not task:
            raise ValueError("Task must not be empty.")
        with self.lock, torch.inference_mode():
            self._reset()
            self.session_id = uuid.uuid4().hex
            self.task = task
            batch = self._observation(images, task)
            started = time.perf_counter()
            actions = self.base_policy.predict_action_chunk(batch)
            elapsed = time.perf_counter() - started
            return self._response(actions, elapsed)

    def infer(self, session_id: str, keyframes: list[dict[str, str]]) -> dict[str, Any]:
        with self.lock, torch.inference_mode():
            if not self.session_id or not hmac.compare_digest(session_id, self.session_id):
                raise ValueError("Unknown or stale session_id; start a new task.")
            expected = self.last_chunk_size // int(self.base_policy.config.action_per_frame)
            if len(keyframes) != expected:
                raise ValueError(
                    f"Expected {expected} keyframes for the {self.last_chunk_size}-action chunk, "
                    f"got {len(keyframes)}."
                )
            raw_observations = []
            for images in keyframes:
                batch = self._observation(images, self.task)
                raw_observations.append(
                    {key: batch[key].detach() for key in self.camera_keys}
                )
            self.base_policy._obs_buffer = raw_observations
            started = time.perf_counter()
            actions = self.base_policy.predict_action_chunk(None)
            elapsed = time.perf_counter() - started
            return self._response(actions, elapsed)

    def close(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if self.session_id and hmac.compare_digest(session_id, self.session_id):
                self._reset()
                self.session_id = None
                self.task = ""
                self.last_chunk_size = 0
        return {"closed": True}

    def health(self) -> dict[str, Any]:
        free_bytes = total_bytes = 0
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "ready": True,
            "device": str(self.base_policy.config.device),
            "camera_keys": list(self.camera_keys),
            "action_channels": list(self.base_policy.config.used_action_channel_ids),
            "action_min": self.action_lower_bound.tolist(),
            "action_max": self.action_upper_bound.tolist(),
            "active_session": self.session_id is not None,
            "cuda_free_gb": free_bytes / 2**30,
            "cuda_total_gb": total_bytes / 2**30,
        }


def make_handler(engine: LingBotRemoteEngine, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LingBotRemote/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.client_address[0]} {fmt % args}", flush=True)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            return hmac.compare_digest(supplied, expected)

        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length.") from exc
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError(f"Request body must be in (0, {MAX_REQUEST_BYTES}] bytes.")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON request body must be an object.")
            return value

        def _dispatch(self, method: str) -> None:
            if not self._authorized():
                self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                if method == "health":
                    response = engine.health()
                elif method == "start":
                    body = self._body()
                    response = engine.start(str(body.get("task", "")), body.get("images", {}))
                elif method == "infer":
                    body = self._body()
                    response = engine.infer(
                        str(body.get("session_id", "")), body.get("keyframes", [])
                    )
                elif method == "close":
                    body = self._body()
                    response = engine.close(str(body.get("session_id", "")))
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._send(HTTPStatus.OK, response)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("health" if self.path == "/health" else "missing")

        def do_POST(self) -> None:  # noqa: N802
            endpoint = self.path.strip("/")
            self._dispatch(endpoint)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--wan-model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.getenv("LINGBOT_REMOTE_TOKEN"))
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-encoder-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--attn-mode", choices=("torch", "flashattn"), default="torch")
    parser.add_argument("--action-inference-steps", type=int, default=50)
    parser.add_argument("--video-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        args.token = _read_token(args.token, args.token_file)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535].")
    if min(args.action_inference_steps, args.video_inference_steps, args.guidance_scale) <= 0:
        parser.error("Inference steps and guidance scale must be positive.")
    return args


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    engine = LingBotRemoteEngine(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, args.token))
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
