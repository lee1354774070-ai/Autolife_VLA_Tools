#!/usr/bin/env python3
"""Deploy a LeRobot 0.6 LingBot-VA LoRA checkpoint on an AutoLife robot.

The trained 16-D action order is the collector's canonical base schema:

    left arm 7, right arm 7, left gripper 1, right gripper 1

Live SHM cameras are mapped by order to the three policy feature keys saved in
the checkpoint (head, left wrist, right wrist). Depth, head joints, and waist
joints are deliberately excluded because this checkpoint was not trained with
them.

Safety defaults are conservative: inference is dry-run unless ``--publish`` is
provided, and publishing stops if any arm joint changes by more than
``--max-arm-delta`` in one command.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR.parent
REPO_DIR = DEPLOY_DIR.parent
for import_dir in (DEPLOY_DIR, REPO_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from pi05.deploy_pi05 import (
    DEFAULT_MAX_IMAGE_DELTA_SEC,
    DEFAULT_SYNC_REFERENCE_CAMERA,
    InteractiveDeploymentSession,
    LocalPI05Deployer,
    env_bool,
)
from lingbot_va.lingbot_va_policy import load_lingbot_policy, unwrap_lingbot_policy


DEFAULT_MODEL_DIR = Path("/home/ubuntu/models/LingBot-VA")
DEFAULT_PHYSICAL_CAMERAS = ("rgbd_head_color", "hand_left", "hand_right")


def _base_policy_config(policy: Any) -> Any:
    """Return the LingBot config whether or not PEFT wraps the base policy."""

    return unwrap_lingbot_policy(policy).config


class LingBotVADeployer(LocalPI05Deployer):
    """Use the shared ROS/SHM controller with LingBot-VA policy semantics."""

    def _print_model_summary(self) -> None:
        assert self.policy is not None
        config = _base_policy_config(self.policy)
        print(f"policy schema: {', '.join(self.schema.names)}")
        print(f"action channels: {config.used_action_channel_ids}")
        print(f"action chunk size: {config.chunk_size} (first chunk returns 12 actions)")
        print(f"attention: {config.attn_mode}, dtype={config.dtype}")
        print(
            "camera mapping: "
            + ", ".join(
                f"{name}->{self.policy_camera_keys[name]}" for name in self.args.camera_names
            )
        )
        print(f"publishing: {'enabled' if not self.args.dry_run else 'DRY-RUN'}")

    def enable(self) -> None:
        """Load and warm the policy without publishing any command."""

        with self._policy_lock:
            if self.policy is None:
                print("Loading LingBot-VA base model and LoRA adapter...")
                try:
                    (
                        self.policy,
                        self.policy_preprocessor,
                        self.policy_postprocessor,
                    ) = load_lingbot_policy(
                        self.args.model_dir,
                        device=self.args.device,
                        base_model=self.args.base_model,
                        wan_model=self.args.wan_model,
                        text_encoder_device=self.args.text_encoder_device,
                        attn_mode=self.args.attn_mode,
                        action_inference_steps=self.args.action_inference_steps,
                        video_inference_steps=self.args.video_inference_steps,
                        guidance_scale=self.args.guidance_scale,
                        offline=self.args.offline,
                    )
                except torch.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "LingBot-VA did not fit in GPU memory. Stop other GPU jobs and retry; "
                        "the robot's 16 GB GPU is close to the model's minimum requirement."
                    ) from exc
                self._print_model_summary()

            if self.model_enabled:
                print("Model is already enabled.")
                return

        self.wait_until_ready()
        q23, _ = self.refresh()
        if q23 is None:
            raise RuntimeError("Joint state disappeared while preparing the model.")

        print("Warming up LingBot-VA without publishing commands...")
        with self._policy_lock:
            self.select_action(q23, self.args.task)
            self._reset_policy_pipeline()
            self.model_enabled = True
        print("Model enabled and ready. Use: start <task text>")

    def build_observation(self, q23: np.ndarray, task: str) -> dict[str, Any]:
        del q23  # This LingBot checkpoint is image/language conditioned, without state input.
        missing = [name for name in self.args.camera_names if name not in self.latest_images]
        if missing:
            raise RuntimeError("Camera observations are not ready: " + ", ".join(missing))
        observation: dict[str, Any] = {"task": task}
        for name in self.args.camera_names:
            image = self.latest_images[name].astype(np.float32, copy=False) / 255.0
            observation[self.policy_camera_keys[name]] = torch.from_numpy(image)
        return observation

    def publish_command(self, command_q23: np.ndarray) -> None:
        """Reject non-finite or discontinuous commands before ROS publication."""

        if not np.isfinite(command_q23).all():
            self.stop_requested = True
            raise RuntimeError("Policy produced a non-finite robot command; publishing stopped.")
        assert self.telemetry is not None
        current_q23, _ = self.telemetry.snapshot()
        if current_q23 is None:
            return
        max_delta = float(np.max(np.abs(command_q23[4:18] - current_q23[4:18])))
        if max_delta > self.args.max_arm_delta:
            self.stop_requested = True
            raise RuntimeError(
                f"Arm command delta {max_delta:.4f} exceeds --max-arm-delta "
                f"{self.args.max_arm_delta:.4f}; publishing stopped."
            )
        super().publish_command(command_q23)


def _checkpoint_camera_keys(model_dir: Path) -> list[str]:
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        keys = config["obs_cam_keys"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read LingBot camera keys from {config_path}: {exc}") from exc
    if not isinstance(keys, list) or len(keys) != 3 or not all(isinstance(k, str) for k in keys):
        raise SystemExit(f"Expected exactly three LingBot obs_cam_keys, got {keys!r}.")
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", default=os.getenv("LINGBOT_BASE_MODEL"))
    parser.add_argument(
        "--wan-model",
        default=os.getenv("LINGBOT_WAN_MODEL", "robbyant/lingbot-va-base"),
        help="Wan repository or local directory containing vae/, text_encoder/, and tokenizer/.",
    )
    parser.add_argument("--task", default="pick up the bottle of water")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Actually publish robot commands; default is dry-run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-encoder-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--attn-mode", choices=("torch", "flashattn"), default="torch")
    parser.add_argument("--action-inference-steps", type=int, default=50)
    parser.add_argument("--video-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=env_bool("HF_HUB_OFFLINE", False))
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-sec", type=float, default=20.0)
    parser.add_argument("--start-delay-sec", type=float, default=3.0)
    parser.add_argument("--max-arm-delta", type=float, default=0.35)
    parser.add_argument("--joints-topic", default=None)
    parser.add_argument("--sync-reference-camera", default=DEFAULT_SYNC_REFERENCE_CAMERA)
    parser.add_argument("--max-image-delta-sec", type=float, default=DEFAULT_MAX_IMAGE_DELTA_SEC)
    parser.add_argument("--max-state-age-sec", type=float, default=0.20)
    parser.add_argument("--max-image-age-sec", type=float, default=0.20)
    parser.add_argument("--gripper-int", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    args.model_dir = args.model_dir.expanduser().resolve()
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model_dir}")
    if args.base_model and Path(args.base_model).expanduser().is_dir():
        args.base_model = str(Path(args.base_model).expanduser().resolve())
    if Path(args.wan_model).expanduser().is_dir():
        args.wan_model = str(Path(args.wan_model).expanduser().resolve())

    args.camera_names = list(DEFAULT_PHYSICAL_CAMERAS)
    args.policy_camera_keys = _checkpoint_camera_keys(args.model_dir)
    args.depth_camera_names = []
    args.with_depth = False
    args.with_head = False
    args.with_waist = False
    args.dry_run = not args.publish
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
        args.hz,
        args.warmup_sec,
        args.max_arm_delta,
        args.max_image_delta_sec,
        args.max_state_age_sec,
        args.max_image_age_sec,
        args.action_inference_steps,
        args.video_inference_steps,
        args.guidance_scale,
        args.print_every,
    )
    if any(value <= 0 for value in positive) or args.max_steps < 0 or args.start_delay_sec < 0:
        raise SystemExit("Timing, inference, safety, and print settings must be positive.")
    if args.sync_reference_camera not in args.camera_names:
        raise SystemExit(
            f"Sync reference {args.sync_reference_camera!r} is not one of {args.camera_names}."
        )
    return args


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    deployer: LingBotVADeployer | None = None
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
        deployer = LingBotVADeployer(args)
        if args.interactive:
            session = InteractiveDeploymentSession(deployer)
            session.run()
        else:
            deployer.run()
    except KeyboardInterrupt:
        pass
    finally:
        if deployer is not None:
            deployer.close()


if __name__ == "__main__":
    main()
