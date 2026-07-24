#!/usr/bin/env python3
"""Run LingBot-VA on a remote GPU server while keeping robot control and safety local."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR.parent
REPO_DIR = DEPLOY_DIR.parent
COLLECTOR_DIR = REPO_DIR / "lerobot_data_collector"
for import_dir in (DEPLOY_DIR, REPO_DIR, COLLECTOR_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from pi05.deploy_pi05 import (
    DEFAULT_MAX_IMAGE_DELTA_SEC,
    DEFAULT_SYNC_REFERENCE_CAMERA,
    InferredAction,
    InteractiveDeploymentSession,
    LocalPI05Deployer,
)
from lingbot_va.lingbot_remote_protocol import (
    LingBotRemoteClient,
    RemoteInferenceError,
    encode_jpeg,
    validate_action_response,
    validate_policy_metadata,
)
from robot_schema import whole_body_from_schema_action
from robot_schema import schema_state_from_whole_body


DEFAULT_TOKEN_FILE = Path("/home/ubuntu/.config/lingbot_remote_token")
DEFAULT_PHYSICAL_CAMERAS = ("rgbd_head_color", "hand_left", "hand_right")
DEFAULT_POLICY_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
EXPECTED_ACTION_CHANNELS = list(range(14, 30))


def _read_token(value: str | None, token_file: Path) -> str:
    if value:
        return value.strip()
    try:
        token = token_file.expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SystemExit(
            f"No remote token was supplied and token file cannot be read: {token_file}: {exc}"
        ) from exc
    if not token:
        raise SystemExit(f"Remote token file is empty: {token_file}")
    return token


class RemoteLingBotDeployer(LocalPI05Deployer):
    """Consume remote action chunks and publish them through local ROS safeguards."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.remote = LingBotRemoteClient(args.server_url, args.token, args.request_timeout_sec)
        self.remote_session_id: str | None = None
        self.pending_task = ""
        self.action_queue: deque[np.ndarray] = deque()
        self.keyframes: list[dict[str, str]] = []
        self.capture_next_observation = False
        self.capture_after_publish = False
        self.chunk_action_index = 0
        self.last_remote_inference_ms = 0.0
        self.action_lower_bound: np.ndarray | None = None
        self.action_upper_bound: np.ndarray | None = None

    def _encoded_images(self) -> dict[str, str]:
        encoded: dict[str, str] = {}
        for physical_name in self.args.camera_names:
            chw_rgb = self.latest_images[physical_name]
            rgb = np.ascontiguousarray(chw_rgb.transpose(1, 2, 0))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            policy_key = self.policy_camera_keys[physical_name]
            encoded[policy_key] = encode_jpeg(
                bgr,
                size=self.args.jpeg_size,
                quality=self.args.jpeg_quality,
            )
        return encoded

    def _clear_rollout(self) -> None:
        self.action_queue.clear()
        self.keyframes.clear()
        self.capture_next_observation = False
        self.capture_after_publish = False
        self.chunk_action_index = 0
        self.last_remote_inference_ms = 0.0

    def enable(self) -> None:
        health = self.remote.health()
        camera_keys, action_channels, lower, upper = validate_policy_metadata(health)
        expected_keys = sorted(self.policy_camera_keys.values())
        if not health.get("ready"):
            raise RuntimeError(f"Remote LingBot server is not ready: {health}")
        if sorted(camera_keys) != expected_keys:
            raise RuntimeError(
                f"Remote cameras {camera_keys} do not match client contract {expected_keys}."
            )
        if action_channels != EXPECTED_ACTION_CHANNELS:
            raise RuntimeError(
                f"Remote action channels {action_channels} do not match "
                f"the robot's expected channels {EXPECTED_ACTION_CHANNELS}."
            )
        margin = np.full(16, self.args.action_range_margin, dtype=np.float32)
        margin[14:] = self.args.gripper_range_margin
        self.action_lower_bound = lower - margin
        self.action_upper_bound = upper + margin
        self.wait_until_ready()
        self.model_enabled = True
        print(
            f"Remote LingBot ready: {self.args.server_url}, device={health.get('device')}, "
            f"cuda_free_gb={health.get('cuda_free_gb', 0):.1f}, "
            f"publishing={'enabled' if not self.args.dry_run else 'DRY-RUN'}"
        )
        print("Use: start <task text>")

    def start_task(self, task: str) -> None:
        task = task.strip()
        if not task:
            raise ValueError("start requires non-empty task text.")
        if not self.model_enabled:
            raise RuntimeError("Remote model is not enabled. Run enable first.")
        if self.remote_session_id:
            self.remote.close(self.remote_session_id)
        self.remote_session_id = None
        self.pending_task = task
        self._clear_rollout()

    def pause_task(self) -> None:
        self.capture_after_publish = False

    def _request_chunk(self) -> None:
        if self.remote_session_id is None:
            response = self.remote.start(self.pending_task, self._encoded_images())
            self.remote_session_id = str(response.get("session_id", ""))
            if not self.remote_session_id:
                raise RemoteInferenceError("Remote start response has no session_id.")
        else:
            response = self.remote.infer(self.remote_session_id, self.keyframes)
        actions = validate_action_response(response)
        self.action_queue.extend(actions)
        self.keyframes.clear()
        self.chunk_action_index = 0
        self.last_remote_inference_ms = float(response.get("inference_ms", 0.0))
        print(
            f"remote chunk: actions={len(actions)} inference_ms={self.last_remote_inference_ms:.1f}"
        )

    def infer_step(self, task: str) -> InferredAction | None:
        loop_started = time.monotonic()
        q23, state_received_sec = self.refresh()
        now = time.time()
        if q23 is None or len(self.latest_images) != len(self.args.camera_names):
            return None
        if now - state_received_sec > self.args.max_state_age_sec:
            return None
        if now - self.latest_image_received_sec > self.args.max_image_age_sec:
            return None
        if not self.model_enabled:
            return None
        if task != self.pending_task:
            raise RuntimeError("Interactive task changed without resetting the remote session.")

        if self.capture_next_observation:
            self.keyframes.append(self._encoded_images())
            self.capture_next_observation = False

        if not self.action_queue:
            self._request_chunk()

        action = np.asarray(self.action_queue.popleft(), dtype=np.float32)
        self.capture_after_publish = (
            (self.chunk_action_index + 1) % self.args.action_per_frame == 0
        )
        self.chunk_action_index += 1
        command_q23 = whole_body_from_schema_action(action, q23, self.schema)
        return InferredAction(
            action=action,
            command_q23=command_q23,
            elapsed_ms=(time.monotonic() - loop_started) * 1000.0,
            inference_ms=self.last_remote_inference_ms,
        )

    def publish_command(self, command_q23: np.ndarray) -> None:
        if not np.isfinite(command_q23).all():
            raise RuntimeError("Remote policy produced a non-finite robot command.")
        assert self.telemetry is not None
        current_q23, _ = self.telemetry.snapshot()
        if current_q23 is None:
            return

        schema_action = schema_state_from_whole_body(command_q23, self.schema)
        if self.action_lower_bound is None or self.action_upper_bound is None:
            raise RuntimeError("Remote checkpoint metadata is unavailable; run enable first.")
        outside = np.flatnonzero(
            (schema_action < self.action_lower_bound)
            | (schema_action > self.action_upper_bound)
        )
        if outside.size:
            index = int(outside[0])
            raise RuntimeError(
                f"Command {self.schema.names[index]}={schema_action[index]:.4f} is outside "
                f"the checkpoint safety envelope "
                f"[{self.action_lower_bound[index]:.4f}, "
                f"{self.action_upper_bound[index]:.4f}]; command rejected."
            )

        gripper_deltas = np.abs(command_q23[18:20] - current_q23[18:20])
        max_gripper_index = int(np.argmax(gripper_deltas))
        max_gripper_delta = float(gripper_deltas[max_gripper_index])
        if max_gripper_delta > self.args.max_gripper_delta:
            name = self.schema.names[14 + max_gripper_index]
            raise RuntimeError(
                f"Gripper command delta {max_gripper_delta:.4f} at {name} exceeds "
                f"--max-gripper-delta {self.args.max_gripper_delta:.4f}; command rejected."
            )

        limited_q23 = command_q23.copy()
        limited_q23[4:18] = current_q23[4:18] + np.clip(
            command_q23[4:18] - current_q23[4:18],
            -self.args.max_arm_step,
            self.args.max_arm_step,
        )
        limited_q23[18:20] = current_q23[18:20] + np.clip(
            command_q23[18:20] - current_q23[18:20],
            -self.args.max_gripper_step,
            self.args.max_gripper_step,
        )
        super().publish_command(limited_q23)
        if self.capture_after_publish:
            self.capture_next_observation = True
        self.capture_after_publish = False

    def close(self) -> None:
        if self.remote_session_id:
            self.remote.close(self.remote_session_id)
            self.remote_session_id = None
        super().close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--server-url", default=os.getenv("LINGBOT_SERVER_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--token", default=os.getenv("LINGBOT_REMOTE_TOKEN"))
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--request-timeout-sec", type=float, default=300.0)
    parser.add_argument("--task", default="pick up the bottle of water")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Actually publish ROS commands; default is dry-run.")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--action-per-frame", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-sec", type=float, default=20.0)
    parser.add_argument("--start-delay-sec", type=float, default=3.0)
    parser.add_argument(
        "--max-arm-step",
        type=float,
        default=4.0,
        help=(
            "Maximum published arm movement in degrees per control period. "
            "Targets within the limit are published unchanged; larger deltas are clipped, not rejected."
        ),
    )
    parser.add_argument("--max-gripper-delta", type=float, default=70.0)
    parser.add_argument("--max-gripper-step", type=float, default=10.0)
    parser.add_argument("--action-range-margin", type=float, default=2.0)
    parser.add_argument("--gripper-range-margin", type=float, default=5.0)
    parser.add_argument("--jpeg-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--joints-topic", default=None)
    parser.add_argument("--sync-reference-camera", default=DEFAULT_SYNC_REFERENCE_CAMERA)
    parser.add_argument("--max-image-delta-sec", type=float, default=DEFAULT_MAX_IMAGE_DELTA_SEC)
    parser.add_argument("--max-state-age-sec", type=float, default=0.20)
    parser.add_argument("--max-image-age-sec", type=float, default=0.20)
    parser.add_argument("--gripper-int", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    # The remote client never loads model weights or checkpoint files. This
    # nonexistent path only satisfies the shared ROS/SHM deployer contract;
    # read_model_contract() treats it as an intentionally absent local model.
    args.model_dir = Path("/__lingbot_remote_checkpoint_lives_on_server__")
    args.token = _read_token(args.token, args.token_file)
    args.camera_names = list(DEFAULT_PHYSICAL_CAMERAS)
    args.policy_camera_keys = list(DEFAULT_POLICY_CAMERA_KEYS)
    # The adapter config retains the source dataset's physical image feature
    # names (including depth), while the remote service consumes the three
    # canonical obs_cam_keys above. The authenticated server validates those
    # keys, so do not apply the stale local image feature dictionary here.
    args.ignore_model_image_contract = True
    args.depth_camera_names = []
    args.with_depth = False
    args.with_head = False
    args.with_waist = False
    args.dry_run = not args.publish
    args.device = "cpu"  # No local policy is loaded; required only by shared argument contracts.
    args.n_action_steps = None
    args.rtc = False
    args.rtc_refresh_steps = None
    args.rtc_execution_horizon = 1
    args.rtc_max_guidance_weight = 1.0
    args.compile_mode = "disabled"
    args.depth_min = 0.05
    args.depth_max = 10.0
    args.depth_shift = 3.5
    args.depth_use_log = True

    positive = (
        args.request_timeout_sec,
        args.hz,
        args.action_per_frame,
        args.warmup_sec,
        args.max_arm_step,
        args.max_gripper_delta,
        args.max_gripper_step,
        args.action_range_margin,
        args.gripper_range_margin,
        args.jpeg_size,
        args.max_image_delta_sec,
        args.max_state_age_sec,
        args.max_image_age_sec,
        args.print_every,
    )
    if any(value <= 0 for value in positive) or args.max_steps < 0 or args.start_delay_sec < 0:
        parser.error("Timing, protocol, safety, and image settings must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100].")
    if args.sync_reference_camera not in args.camera_names:
        parser.error(f"Invalid sync reference: {args.sync_reference_camera}.")
    return args


def main() -> None:
    args = parse_args()
    deployer: RemoteLingBotDeployer | None = None
    session: InteractiveDeploymentSession | None = None

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        if session is not None:
            session.request_exit()
        elif deployer is not None:
            deployer.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        deployer = RemoteLingBotDeployer(args)
        if args.interactive:
            session = InteractiveDeploymentSession(deployer)
            session.run()
        else:
            deployer.run()
    except (KeyboardInterrupt, RemoteInferenceError):
        pass
    finally:
        if deployer is not None:
            deployer.close()


if __name__ == "__main__":
    main()
