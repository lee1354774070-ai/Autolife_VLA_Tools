#!/usr/bin/env python3
"""Run a local LeRobot PI0.5 policy on an AutoLife robot.

The collector and this deployer use one canonical policy layout:

    base:  left arm 7, right arm 7, left gripper 1, right gripper 1
    head:  neck roll, pitch, yaw (optional, appended after base)
    waist: leg ankle, knee, waist pitch, yaw (optional, appended after head)

The ROS controller has a different physical whole-body layout:

    leg/waist 4, left arm 7, right arm 7, grippers 2, neck 3

The conversion is explicit below. Disabled head/waist groups are copied from
the latest measured joint state, so the policy cannot move those groups.

This file intentionally uses the collector's direct SHM reader instead of the
ROS image bridge. That removes an extra serialization hop while retaining the
same metadata-consistency checks used during recording.

By default the policy receives the three RGB features used by the current
collector: ``rgbd_head_color``, ``hand_left``, and ``hand_right``. Depth and
the optional head/waist joint groups are enabled explicitly with command-line
switches so one deployer can serve all matching PI0.5 checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import torch
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String as StringMsg


# Allow this script to be copied as a complete /deploy directory while still
# reusing shared deployment helpers, the camera ABI, and the canonical schema.
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR.parent
REPO_DIR = DEPLOY_DIR.parent
COLLECTOR_DIR = REPO_DIR / "lerobot_data_collector"
for import_dir in (DEPLOY_DIR, REPO_DIR, COLLECTOR_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from camera_config import (  # noqa: E402
    CAMERA_SPECS,
    DEFAULT_DEPTH_CAMERA_NAMES,
    DEFAULT_RGB_CAMERA_NAMES,
    CameraSpec,
)
from cli_help import show_requested_parameter_help  # noqa: E402
from common.session_control import SessionControl  # noqa: E402
from robot_schema import (  # noqa: E402
    RobotSchema,
    build_robot_schema,
    parse_whole_body_state,
    schema_state_from_whole_body,
    whole_body_from_schema_action,
)
from shm_camera import frame_to_hwc, read_shm_frame, read_shm_metadata, shm_timestamp_sec  # noqa: E402


DEFAULT_RGB_SHAPE = (3, 480, 640)
DEFAULT_DEPTH_SHAPE = (1, 480, 640)
DEFAULT_SYNC_REFERENCE_CAMERA = "hand_left"
DEFAULT_MAX_IMAGE_DELTA_SEC = 0.04
DEFAULT_DEPTH_MIN = 0.05
DEFAULT_DEPTH_MAX = 10.0
DEFAULT_DEPTH_SHIFT = 3.5


def env_bool(name: str, default: bool) -> bool:
    """Parse the collector-style 0/1 environment switches."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_robot_id() -> str:
    """Use ROBOT_ID first, then the numeric suffix in the host name."""

    robot_id = os.getenv("ROBOT_ID")
    if robot_id:
        return robot_id
    match = re.search(r"-(\d+)$", os.uname().nodename)
    return match.group(1) if match else "283"


def topic_id() -> str:
    return f"{os.getenv('ROS_DOMAIN_ID', '0')}_{default_robot_id()}"


def image_to_policy_chw(image: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a BGR SHM image into a normalized RGB CHW policy input."""

    if image.dtype != np.uint8:
        image = np.asarray(image)
        minimum, maximum = float(image.min()), float(image.max())
        if maximum > minimum:
            image = ((image - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected a BGR image with 3 channels, got shape {image.shape}.")
    image = image[..., :3]
    image = cv2.resize(image, (target_shape[2], target_shape[1]), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    if chw.shape != target_shape:
        raise ValueError(f"Image has shape {chw.shape}, expected {target_shape}.")
    return chw


def depth_to_policy_chw(
    depth_hwc: np.ndarray,
    target_shape: tuple[int, int, int],
    depth_min: float,
    depth_max: float,
    depth_shift: float,
    depth_use_log: bool,
) -> np.ndarray:
    """Quantize live millimetre depth like LeRobot's native depth video path.

    The collector stores physical depth as uint16 millimetres. LeRobot 0.6
    quantizes that stream to uint12 codes for video, so deployment must apply
    the same conversion before passing a depth feature to the policy.
    """

    depth = np.asarray(depth_hwc)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a single-channel depth map, got shape {depth.shape}.")
    depth = cv2.resize(depth, (target_shape[2], target_shape[1]), interpolation=cv2.INTER_NEAREST)
    try:
        from lerobot.datasets.video_utils import quantize_depth

        quantized = quantize_depth(
            depth.astype(np.uint16, copy=False),
            depth_min=depth_min,
            depth_max=depth_max,
            shift=depth_shift,
            use_log=depth_use_log,
            video_backend=None,
            input_unit="mm",
        )
        quantized = np.asarray(quantized, dtype=np.float32)
    except ImportError as exc:
        raise RuntimeError("Depth deployment requires LeRobot 0.6 video_utils.quantize_depth.") from exc
    result = np.ascontiguousarray(quantized[None, ...])
    if result.shape != target_shape:
        raise ValueError(f"Depth has shape {result.shape}, expected {target_shape}.")
    return result


def configure_policy_compilation(config: Any, compile_mode: str) -> None:
    """Apply a deployment-time torch.compile override to a saved config."""

    if compile_mode == "checkpoint":
        return
    if compile_mode == "disabled":
        config.compile_model = False
        return
    config.compile_model = True
    config.compile_mode = compile_mode


def configure_policy_runtime(
    config: Any,
    n_action_steps: int | None,
    rtc_enabled: bool,
    rtc_execution_horizon: int,
    rtc_max_guidance_weight: float,
) -> None:
    """Apply action-chunk and RTC overrides before constructing PI0.5."""

    chunk_size = int(config.chunk_size)
    if n_action_steps is not None:
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError(
                f"n_action_steps must be in [1, {chunk_size}], got {n_action_steps}."
            )
        config.n_action_steps = n_action_steps

    if not rtc_enabled:
        config.rtc_config = None
        return
    if not 1 <= rtc_execution_horizon <= chunk_size:
        raise ValueError(
            f"RTC execution horizon must be in [1, {chunk_size}], "
            f"got {rtc_execution_horizon}."
        )
    try:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
    except ImportError as exc:
        raise RuntimeError("RTC requires LeRobot 0.6 or newer.") from exc
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=rtc_execution_horizon,
        max_guidance_weight=rtc_max_guidance_weight,
    )


def load_policy(
    model_dir: Path,
    device: str,
    tokenizer_dir: Path | None = None,
    compile_mode: str = "checkpoint",
    n_action_steps: int | None = None,
    rtc_enabled: bool = False,
    rtc_execution_horizon: int = 10,
    rtc_max_guidance_weight: float = 10.0,
):
    """Load a PI0.5 policy together with its saved I/O processors."""

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError as exc:
        raise SystemExit(
            "Cannot import PI05Policy. Activate the LeRobot environment used for training."
        ) from exc

    config = PreTrainedConfig.from_pretrained(str(model_dir))
    config.device = device
    configure_policy_compilation(config, compile_mode)
    configure_policy_runtime(
        config,
        n_action_steps,
        rtc_enabled,
        rtc_execution_horizon,
        rtc_max_guidance_weight,
    )
    policy = PI05Policy.from_pretrained(str(model_dir), config=config)
    policy = policy.to(device).eval()
    preprocessor_overrides: dict[str, dict[str, str]] = {}
    if tokenizer_dir is not None:
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_dir}")
        preprocessor_overrides = {
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides=preprocessor_overrides,
    )
    if hasattr(policy, "reset"):
        policy.reset()
    return policy, preprocessor, postprocessor


def read_model_contract(
    model_dir: Path,
) -> tuple[int | None, int | None, dict[str, tuple[int, ...]], list[str] | None]:
    """Read state/action/image features before touching the robot controller."""

    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None, None, {}, None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        input_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
        action_shape = config.get("output_features", {}).get("action", {}).get("shape")
        input_features = config.get("input_features", {})
        image_shapes = {
            key: tuple(int(value) for value in feature.get("shape", []))
            for key, feature in input_features.items()
            if key.startswith("observation.images.") and feature.get("shape")
        }
        action_feature_names = config.get("action_feature_names")
        if not isinstance(action_feature_names, list):
            action_feature_names = None
        state_dim = int(input_shape[0]) if isinstance(input_shape, list) and input_shape else None
        action_dim = int(action_shape[0]) if isinstance(action_shape, list) and action_shape else None
        return state_dim, action_dim, image_shapes, action_feature_names
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse model contract: {config_path}: {exc}") from exc


def normalize_policy_image_shape(shape: tuple[int, ...], camera_name: str) -> tuple[int, int, int]:
    """Normalize a LeRobot visual feature shape to channel-first order."""

    if len(shape) != 3:
        raise ValueError(f"Model feature observation.images.{camera_name} must be 3-D, got {shape}.")
    if shape[0] in (1, 3):
        return shape
    if shape[-1] in (1, 3):
        return (shape[-1], shape[0], shape[1])
    raise ValueError(f"Model feature observation.images.{camera_name} has no 1/3-channel dimension: {shape}.")


class JointTelemetryNode(Node):
    """Keep the newest complete physical q23 joint state from ROS2."""

    def __init__(self, joints_topic: str) -> None:
        super().__init__("pi05_local_deploy_telemetry")
        self._lock = threading.Lock()
        self.latest_q23: np.ndarray | None = None
        self.latest_received_sec = 0.0
        self.subscription = self.create_subscription(
            StringMsg, joints_topic, self._on_joints, 10
        )
        self.get_logger().info(f"subscribing to joint state: {joints_topic}")

    def _on_joints(self, message: StringMsg) -> None:
        try:
            packet = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(packet, dict):
            return
        q23 = parse_whole_body_state(packet)
        if q23 is None:
            return
        with self._lock:
            self.latest_q23 = q23
            self.latest_received_sec = time.time()

    def snapshot(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            q23 = None if self.latest_q23 is None else self.latest_q23.copy()
            return q23, self.latest_received_sec


class AutoLifeCommandNode(Node):
    """Publish physical q23 commands on the same topics as the collector."""

    def __init__(self) -> None:
        super().__init__("pi05_local_deploy_command")
        tid = topic_id()
        body_topic = f"/topic_arm_whole_body_target_joints_position_{tid}"
        gripper_topic = f"/topic_arm_gripper_target_joints_position_{tid}"
        self.body_publisher = self.create_publisher(StringMsg, body_topic, 10)
        self.gripper_publisher = self.create_publisher(StringMsg, gripper_topic, 10)
        self.get_logger().info(f"publishing body commands: {body_topic}")
        self.get_logger().info(f"publishing gripper commands: {gripper_topic}")

    @staticmethod
    def publishable_body(q23: np.ndarray) -> dict[str, list[float]]:
        return {
            "leg_waist_target_joints_position": q23[:4].astype(float).tolist(),
            "left_arm_target_joints_position": q23[4:11].astype(float).tolist(),
            "right_arm_target_joints_position": q23[11:18].astype(float).tolist(),
            "neck_target_joints_position": q23[20:23].astype(float).tolist(),
        }

    @staticmethod
    def publishable_grippers(q23: np.ndarray, as_int: bool) -> dict[str, list[float] | list[int]]:
        if as_int:
            return {
                "left_gripper_target_joints_position": [int(round(float(q23[18])))],
                "right_gripper_target_joints_position": [int(round(float(q23[19])))],
            }
        return {
            "left_gripper_target_joints_position": [float(q23[18])],
            "right_gripper_target_joints_position": [float(q23[19])],
        }

    def publish_action(self, q23: np.ndarray, gripper_int: bool) -> None:
        body = StringMsg()
        body.data = json.dumps(self.publishable_body(q23), separators=(",", ":"))
        self.body_publisher.publish(body)

        grippers = StringMsg()
        grippers.data = json.dumps(
            self.publishable_grippers(q23, gripper_int), separators=(",", ":")
        )
        self.gripper_publisher.publish(grippers)


@dataclass(frozen=True)
class ShmImage:
    """One decoded source image plus the timestamp used for multi-camera sync."""

    image_hwc: np.ndarray
    stamp_sec: float
    received_sec: float
    is_depth: bool


@dataclass(frozen=True)
class InferredAction:
    """One policy result kept separate from the side-effect of publishing it."""

    action: np.ndarray
    command_q23: np.ndarray
    elapsed_ms: float
    inference_ms: float | None = None


class DirectShmCamera:
    """Read only new, internally consistent frames from one camera's SHM ABI."""

    def __init__(self, spec: CameraSpec, is_depth: bool) -> None:
        self.spec = spec
        self.is_depth = is_depth
        self.last_timestamp_ns: int | None = None

    def read_latest(self, metadata: tuple[int, ...]) -> ShmImage | None:
        if self.last_timestamp_ns == metadata[0]:
            return None
        frame = read_shm_frame(self.spec, metadata)
        if frame is None or frame.timestamp_ns != metadata[0]:
            return None
        self.last_timestamp_ns = frame.timestamp_ns
        received_sec = time.time()
        return ShmImage(
            image_hwc=frame_to_hwc(frame, self.is_depth),
            stamp_sec=shm_timestamp_sec(frame.timestamp_ns, received_sec),
            received_sec=received_sec,
            is_depth=self.is_depth,
        )


class DirectShmCameraSet:
    """Batch-poll selected cameras and keep their newest timestamped frames."""

    def __init__(self, camera_names: tuple[str, ...], depth_names: set[str]) -> None:
        self.camera_names = camera_names
        self.cameras = {
            name: DirectShmCamera(CAMERA_SPECS[name], name in depth_names)
            for name in camera_names
        }
        self.latest: dict[str, ShmImage] = {}

    def refresh(self) -> None:
        # Snapshot all metadata before copying any large image buffer, matching
        # the collector's metadata-batch read strategy.
        metadata_by_name = {}
        for name in self.camera_names:
            metadata = read_shm_metadata(self.cameras[name].spec)
            if metadata is not None:
                metadata_by_name[name] = metadata
        for name in self.camera_names:
            camera = self.cameras[name]
            metadata = metadata_by_name.get(name)
            if metadata is None or camera.last_timestamp_ns == metadata[0]:
                continue
            image = camera.read_latest(metadata)
            if image is not None:
                self.latest[name] = image

    def synchronized_latest(
        self,
        reference_name: str,
        max_delta_sec: float,
        max_age_sec: float,
    ) -> dict[str, ShmImage] | None:
        if reference_name not in self.latest or any(name not in self.latest for name in self.camera_names):
            return None
        reference = self.latest[reference_name]
        now = time.time()
        selected: dict[str, ShmImage] = {}
        for name in self.camera_names:
            image = self.latest[name]
            if now - image.received_sec > max_age_sec:
                return None
            if abs(image.stamp_sec - reference.stamp_sec) > max_delta_sec:
                return None
            selected[name] = image
        return selected


def _normalize_rtc_prefix(actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Keep RTC prefix shapes stable so torch.compile does not recompile each refresh."""

    if actions.ndim != 2:
        raise ValueError(f"RTC prefix must have shape [steps, actions], got {tuple(actions.shape)}.")
    if len(actions) >= target_steps:
        return actions[:target_steps]
    padded = torch.zeros(
        (target_steps, actions.shape[1]),
        dtype=actions.dtype,
        device=actions.device,
    )
    padded[: len(actions)] = actions
    return padded


class RTCActionEngine:
    """Run PI0.5 chunk inference asynchronously and merge overlapping chunks.

    The main robot loop only supplies the newest observation and consumes one
    queued action. This worker owns policy inference, so camera/state polling
    and command publication can continue while the next chunk is generated.
    """

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        policy_lock: threading.RLock,
        hz: float,
        refresh_steps: int,
    ) -> None:
        try:
            from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
            from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep
        except ImportError as exc:
            raise RuntimeError("RTC requires the LeRobot 0.6 RTC modules.") from exc

        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.policy_lock = policy_lock
        self.config = policy.config.rtc_config
        self.refresh_steps = refresh_steps
        self.period_sec = 1.0 / hz
        self.queue = ActionQueue(self.config)
        self.latencies = LatencyTracker()
        self.reanchor_relative_rtc_prefix = reanchor_relative_rtc_prefix

        steps = getattr(preprocessor, "steps", ())
        self.relative_step = next(
            (step for step in steps if isinstance(step, RelativeActionsProcessorStep) and step.enabled),
            None,
        )
        self.normalizer_step = next(
            (step for step in steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )
        if self.relative_step is not None and self.relative_step.action_names is None:
            names = getattr(policy.config, "action_feature_names", None)
            if names:
                self.relative_step.action_names = list(names)

        self._state_lock = threading.Lock()
        self._latest_observation: dict[str, Any] | None = None
        self._generation = 0
        self._error: Exception | None = None
        self._inference_count = 0
        self._last_inference_ms = 0.0
        self._active = threading.Event()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._run, name="PI05-RTC", daemon=True)
        self._thread.start()

    @property
    def last_inference_ms(self) -> float:
        with self._state_lock:
            return self._last_inference_ms

    @property
    def inference_count(self) -> int:
        with self._state_lock:
            return self._inference_count

    def _reset_pipeline(self) -> None:
        self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            if hasattr(processor, "reset"):
                processor.reset()

    def start_task(self) -> None:
        """Invalidate old work, reset all chunk state, and resume inference."""

        self._active.clear()
        with self._state_lock:
            self._generation += 1
            self._latest_observation = None
            self._error = None
        with self.policy_lock:
            self._reset_pipeline()
            self.queue.clear()
            self.latencies.reset()
        self._active.set()

    def pause(self) -> None:
        """Stop generating and discard actions based on the paused observation."""

        self._active.clear()
        with self._state_lock:
            self._generation += 1
            self._latest_observation = None
        self.queue.clear()

    def stop(self) -> None:
        self.pause()
        self._shutdown.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            print("Warning: RTC inference thread did not stop within 5 seconds.")

    def notify_observation(self, observation: dict[str, Any]) -> None:
        with self._state_lock:
            self._latest_observation = observation

    def get_action(self) -> torch.Tensor | None:
        with self._state_lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"RTC background inference failed: {error}") from error
        return self.queue.get()

    def _postprocess_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        processed = self.postprocessor(actions)
        if isinstance(processed, dict):
            processed = processed.get("action")
        if not isinstance(processed, torch.Tensor):
            processed = torch.as_tensor(processed)
        if processed.ndim == 3 and processed.shape[0] == 1:
            processed = processed.squeeze(0)
        if processed.ndim != 2:
            raise ValueError(f"RTC postprocessor returned shape {tuple(processed.shape)}.")
        return processed

    def _run(self) -> None:
        while not self._shutdown.is_set():
            if not self._active.is_set():
                time.sleep(0.01)
                continue
            with self._state_lock:
                observation = self._latest_observation
                generation = self._generation
            needs_chunk = self.queue.empty() or self.queue.get_action_index() >= self.refresh_steps
            if observation is None or not needs_chunk:
                time.sleep(0.005)
                continue

            try:
                started = time.perf_counter()
                action_index = self.queue.get_action_index()
                previous = self.queue.get_left_over()
                predicted_delay = math.ceil((self.latencies.max() or 0.0) / self.period_sec)

                with self.policy_lock, torch.inference_mode():
                    batch = self.preprocessor(observation)
                    if previous is not None and self.relative_step is not None:
                        raw_state = self.relative_step.get_cached_state()
                        previous_absolute = self.queue.get_processed_left_over()
                        if raw_state is not None and previous_absolute is not None:
                            previous = self.reanchor_relative_rtc_prefix(
                                prev_actions_absolute=previous_absolute,
                                current_state=raw_state,
                                relative_step=self.relative_step,
                                normalizer_step=self.normalizer_step,
                                policy_device=next(self.policy.parameters()).device,
                            )
                    if previous is not None:
                        previous = _normalize_rtc_prefix(
                            previous,
                            self.config.execution_horizon,
                        )
                    actions = self.policy.predict_action_chunk(
                        batch,
                        inference_delay=predicted_delay,
                        prev_chunk_left_over=previous,
                    )
                    original = actions.squeeze(0).clone()
                    processed = self._postprocess_chunk(actions)

                elapsed = time.perf_counter() - started
                self.latencies.add(elapsed)
                # Count actions actually consumed while inference was running.
                # This is more accurate than dropping actions from wall time
                # when the queue was already empty and the robot was waiting.
                actual_delay = max(0, self.queue.get_action_index() - action_index)
                with self._state_lock:
                    stale = generation != self._generation or not self._active.is_set()
                if stale:
                    continue
                self.queue.merge(original, processed, actual_delay, action_index)
                with self._state_lock:
                    self._inference_count += 1
                    self._last_inference_ms = elapsed * 1000.0
            except Exception as exc:
                with self._state_lock:
                    self._error = exc
                self._active.clear()
                print(f"RTC inference stopped: {exc}")


class LocalPI05Deployer:
    """Coordinate state/image snapshots, policy inference, and robot commands.

    ROS2 and SHM readers are initialized when this object is created. The PI0.5
    model itself is deliberately loaded only by :meth:`enable`, allowing the
    interactive console to keep a warmed policy in GPU memory between tasks.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False
        self.schema = build_robot_schema(args.with_head, args.with_waist)
        self.joints_topic = args.joints_topic or (
            f"/topic_arm_whole_body_and_gripper_current_joints_status_{topic_id()}"
        )
        self.telemetry: JointTelemetryNode | None = None
        self.command_node: AutoLifeCommandNode | None = None
        self.executor: MultiThreadedExecutor | None = None
        self.spin_thread: threading.Thread | None = None
        self._policy_lock = threading.RLock()
        self.policy: Any | None = None
        self.policy_preprocessor: Any | None = None
        self.policy_postprocessor: Any | None = None
        self.rtc_engine: RTCActionEngine | None = None
        self.model_enabled = False

        contract_state, contract_action, contract_images, contract_action_names = read_model_contract(args.model_dir)
        if getattr(args, "ignore_model_image_contract", False):
            contract_images = {}
        expected_dim = self.schema.size
        if contract_state is not None and contract_state != expected_dim:
            raise ValueError(
                f"Model expects observation.state={contract_state} dims, but the selected "
                f"head/waist switches produce {expected_dim}. Use the switches used during training."
            )
        if contract_action is not None and contract_action != expected_dim:
            raise ValueError(
                f"Model action has {contract_action} dims, but the selected schema needs {expected_dim}."
            )
        expected_camera_names = tuple(args.camera_names)
        policy_camera_keys = tuple(
            getattr(
                args,
                "policy_camera_keys",
                tuple(f"observation.images.{name}" for name in expected_camera_names),
            )
        )
        if len(policy_camera_keys) != len(expected_camera_names):
            raise ValueError(
                "Physical camera names and policy camera keys must have the same length: "
                f"{len(expected_camera_names)} != {len(policy_camera_keys)}."
            )
        expected_image_keys = set(policy_camera_keys)
        if contract_images:
            model_image_keys = set(contract_images)
            if model_image_keys != expected_image_keys:
                missing = sorted(expected_image_keys - model_image_keys)
                unexpected = sorted(model_image_keys - expected_image_keys)
                details = []
                if missing:
                    details.append("missing=" + ",".join(missing))
                if unexpected:
                    details.append("unexpected=" + ",".join(unexpected))
                raise ValueError(
                    "Selected cameras do not match model input features ("
                    + "; ".join(details)
                    + "). Use the cameras used during training or a matching checkpoint."
                )
            self.image_shapes = {
                camera_name: normalize_policy_image_shape(contract_images[policy_key], policy_key)
                for camera_name, policy_key in zip(
                    expected_camera_names, policy_camera_keys, strict=True
                )
            }
        else:
            self.image_shapes = {
                name: DEFAULT_DEPTH_SHAPE if name in args.depth_camera_names else DEFAULT_RGB_SHAPE
                for name in expected_camera_names
            }
        if contract_action_names and tuple(contract_action_names[:expected_dim]) != self.schema.names:
            print(
                "Warning: model action labels differ from collector canonical names; "
                "deployment preserves the trained array order."
            )

        rclpy.init(args=None)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.telemetry = JointTelemetryNode(self.joints_topic)
        self.command_node = AutoLifeCommandNode()
        self.executor.add_node(self.telemetry)
        self.executor.add_node(self.command_node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.camera_set = DirectShmCameraSet(tuple(args.camera_names), set(args.depth_camera_names))
        self.latest_images: dict[str, np.ndarray] = {}
        self.latest_image_received_sec = 0.0
        self.contract_action_dim = contract_action
        self.policy_camera_keys = dict(
            zip(expected_camera_names, policy_camera_keys, strict=True)
        )

    def _print_model_summary(self) -> None:
        print(f"policy schema: {', '.join(self.schema.names)}")
        print(f"policy state/action dimensions: {self.schema.size}/{self.contract_action_dim or 'unknown'}")
        camera_inputs = ", ".join(
            f"{name}->{self.policy_camera_keys[name]}" for name in self.args.camera_names
        )
        print(f"camera inputs: {camera_inputs}")
        print(f"camera source: direct SHM, reference={self.args.sync_reference_camera}")
        assert self.policy is not None
        print(f"action chunk: size={self.policy.config.chunk_size}, n_action_steps={self.policy.config.n_action_steps}")
        if self.args.rtc:
            print(
                f"RTC: enabled, execution_horizon={self.args.rtc_execution_horizon}, "
                f"refresh_steps={self._rtc_refresh_steps()}"
            )
        else:
            print("RTC: disabled")

    def _rtc_refresh_steps(self) -> int:
        """Resolve the upstream default queue threshold to consumed steps."""

        assert self.policy is not None
        chunk_size = int(self.policy.config.chunk_size)
        # LeRobot's RTCInferenceEngine defaults to refreshing when 30 actions
        # remain. For a 50-step PI0.5 chunk, that is after 20 consumed steps.
        steps = self.args.rtc_refresh_steps
        steps = max(1, chunk_size - 30) if steps is None else steps
        if not 1 <= steps <= chunk_size:
            raise ValueError(f"rtc_refresh_steps must be in [1, {chunk_size}], got {steps}.")
        return steps

    def _reset_policy_pipeline(self) -> None:
        assert self.policy is not None
        self.policy.reset()
        for processor in (self.policy_preprocessor, self.policy_postprocessor):
            if hasattr(processor, "reset"):
                processor.reset()

    def enable(self) -> None:
        """Load and compile the model once without publishing a robot command."""

        with self._policy_lock:
            if self.policy is None:
                print("Loading PI0.5 policy into memory...")
                (
                    self.policy,
                    self.policy_preprocessor,
                    self.policy_postprocessor,
                ) = load_policy(
                    self.args.model_dir,
                    self.args.device,
                    self.args.tokenizer_dir,
                    self.args.compile_mode,
                    self.args.n_action_steps,
                    self.args.rtc,
                    self.args.rtc_execution_horizon,
                    self.args.rtc_max_guidance_weight,
                )
                self._print_model_summary()

            if self.model_enabled:
                print("Model is already enabled.")
                return

        self.wait_until_ready()
        q23, _ = self.refresh()
        if q23 is None:
            raise RuntimeError("Joint state disappeared while preparing the model.")

        # Run one complete forward pass to populate CUDA and Torch compile
        # caches. It intentionally never reaches publish_action(). Resetting
        # afterwards discards the warm-up action chunk before a real task.
        print("Warming up PI0.5 policy without publishing commands...")
        with self._policy_lock:
            assert self.policy is not None
            if self.args.rtc:
                assert self.policy_preprocessor is not None
                assert self.policy_postprocessor is not None
                observation = self.build_observation(q23, self.args.task)
                with torch.inference_mode():
                    chunk = self.policy.predict_action_chunk(
                        self.policy_preprocessor(observation),
                        inference_delay=0,
                        prev_chunk_left_over=None,
                    )
                    self.policy_postprocessor(chunk)
            else:
                self.select_action(q23, self.args.task)
            self._reset_policy_pipeline()
            if self.args.rtc:
                self.rtc_engine = RTCActionEngine(
                    self.policy,
                    self.policy_preprocessor,
                    self.policy_postprocessor,
                    self._policy_lock,
                    self.args.hz,
                    self._rtc_refresh_steps(),
                )
            self.model_enabled = True
        print("Model enabled and ready. Use: start <task text>")

    def start_task(self, task: str) -> None:
        """Clear a previous action chunk before a new natural-language task."""

        task = task.strip()
        if not task:
            raise ValueError("start requires task text, for example: start pick up the water bottle")
        if not self.model_enabled or self.policy is None:
            raise RuntimeError("Model is not enabled. Run enable first.")
        if self.rtc_engine is not None:
            self.rtc_engine.start_task()
            return
        with self._policy_lock:
            self._reset_policy_pipeline()

    def pause_task(self) -> None:
        """Pause RTC immediately and invalidate its queued or in-flight actions."""

        if self.rtc_engine is not None:
            self.rtc_engine.pause()

    def infer_step(self, task: str) -> InferredAction | None:
        """Infer one fresh action without publishing it to the robot."""

        loop_started = time.monotonic()
        q23, state_received_sec = self.refresh()
        now = time.time()
        if q23 is None or len(self.latest_images) != len(self.args.camera_names):
            return None
        if now - state_received_sec > self.args.max_state_age_sec:
            return None
        if now - self.latest_image_received_sec > self.args.max_image_age_sec:
            return None

        if not self.model_enabled or self.policy is None:
            return None
        inference_ms = None
        if self.rtc_engine is not None:
            self.rtc_engine.notify_observation(self.build_observation(q23, task))
            queued_action = self.rtc_engine.get_action()
            if queued_action is None:
                return None
            action = self._action_array(queued_action)
            inference_ms = self.rtc_engine.last_inference_ms
        else:
            with self._policy_lock:
                action = self.select_action(q23, task)
        command_q23 = whole_body_from_schema_action(action, q23, self.schema)
        return InferredAction(
            action,
            command_q23,
            (time.monotonic() - loop_started) * 1000.0,
            inference_ms,
        )

    def publish_command(self, command_q23: np.ndarray) -> None:
        """Publish a command already approved by the session state machine."""

        if self.args.dry_run:
            return
        assert self.command_node is not None
        self.command_node.publish_action(command_q23, self.args.gripper_int)

    def close(self) -> None:
        self.stop_requested = True
        if self.rtc_engine is not None:
            self.rtc_engine.stop()
            self.rtc_engine = None
        if self.executor is not None:
            try:
                self.executor.shutdown()
            except Exception:
                pass
        for node in (self.telemetry, self.command_node):
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:
                    pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        with self._policy_lock:
            self.model_enabled = False
            self.policy_postprocessor = None
            self.policy_preprocessor = None
            self.policy = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def refresh(self) -> tuple[np.ndarray | None, float]:
        assert self.telemetry is not None
        q23, state_received_sec = self.telemetry.snapshot()
        self.camera_set.refresh()
        synchronized = self.camera_set.synchronized_latest(
            self.args.sync_reference_camera,
            self.args.max_image_delta_sec,
            self.args.max_image_age_sec,
        )
        if synchronized is not None:
            self.latest_images = {}
            for name, image in synchronized.items():
                shape = self.image_shapes[name]
                if image.is_depth:
                    self.latest_images[name] = depth_to_policy_chw(
                        image.image_hwc,
                        shape,
                        self.args.depth_min,
                        self.args.depth_max,
                        self.args.depth_shift,
                        self.args.depth_use_log,
                    )
                else:
                    self.latest_images[name] = image_to_policy_chw(image.image_hwc, shape)
            # Track the oldest camera in the synchronized observation. Using
            # the newest receive time could let one stale camera survive the
            # outer freshness check longer than intended.
            self.latest_image_received_sec = min(image.received_sec for image in synchronized.values())
        return q23, state_received_sec

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.args.warmup_sec
        while not self.stop_requested and time.monotonic() < deadline:
            q23, state_received_sec = self.refresh()
            image_ready = len(self.latest_images) == len(self.args.camera_names)
            state_ready = q23 is not None
            if state_ready and image_ready:
                print("Ready: joint state and camera frame are available.")
                return
            time.sleep(0.01)
        missing = []
        if self.telemetry is None or self.telemetry.latest_q23 is None:
            missing.append(f"joint state on {self.joints_topic}")
        missing_cameras = [name for name in self.args.camera_names if name not in self.latest_images]
        if missing_cameras:
            missing.append("cameras " + ", ".join(missing_cameras))
        raise TimeoutError(f"Timed out waiting for: {', '.join(missing)}")

    def build_observation(self, q23: np.ndarray, task: str) -> dict[str, Any]:
        missing = [name for name in self.args.camera_names if name not in self.latest_images]
        if missing:
            raise RuntimeError("Camera observations are not ready: " + ", ".join(missing))
        state = schema_state_from_whole_body(q23, self.schema)
        observation: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": task,
        }
        for name in self.args.camera_names:
            image = self.latest_images[name]
            if name not in self.args.depth_camera_names:
                image = image.astype(np.float32, copy=False) / 255.0
            observation[self.policy_camera_keys[name]] = torch.from_numpy(image)
        return observation

    def select_action(self, q23: np.ndarray, task: str) -> np.ndarray:
        assert self.policy is not None
        assert self.policy_preprocessor is not None
        assert self.policy_postprocessor is not None
        observation = self.build_observation(q23, task)
        with torch.inference_mode():
            # PI0.5 requires the saved processor pipeline: it normalizes state,
            # tokenizes ``task`` into observation.language.* fields, and moves
            # tensors to the policy device. The postprocessor restores physical
            # action values using the checkpoint's training statistics.
            action = self.policy.select_action(self.policy_preprocessor(observation))
            action = self.policy_postprocessor(action)
        return self._action_array(action)

    def _action_array(self, action: Any) -> np.ndarray:
        """Convert one postprocessed LeRobot action to the selected robot schema."""

        if isinstance(action, dict):
            action = action.get("action")
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action is None:
            raise ValueError("Policy returned no action.")
        array = np.asarray(action, dtype=np.float32)
        expected_dim = self.schema.size
        # PI0.5 may return a batch or an action chunk. The first action in the
        # chunk is the command for the current observation.
        if array.ndim > 1:
            if array.shape[-1] < expected_dim:
                raise ValueError(f"Policy action shape {array.shape} is too short.")
            array = array.reshape(-1, array.shape[-1])[0]
        array = array.reshape(-1)
        if array.size < expected_dim:
            raise ValueError(f"Policy returned {array.size} dims, expected {expected_dim}.")
        return array[:expected_dim]

    def run(self) -> None:
        self.enable()
        self.start_task(self.args.task)
        period_sec = 1.0 / max(self.args.hz, 1e-6)
        print(
            f"Inference started: task={self.args.task!r}, hz={self.args.hz}, "
            f"head={'on' if self.args.with_head else 'off'}, "
            f"waist={'on' if self.args.with_waist else 'off'}, dry_run={self.args.dry_run}"
        )
        if self.args.start_delay_sec > 0:
            print(f"Starting in {self.args.start_delay_sec:.1f}s. Press Ctrl-C to cancel.")
            time.sleep(self.args.start_delay_sec)

        next_deadline = time.monotonic()
        step = 0
        while not self.stop_requested and (self.args.max_steps <= 0 or step < self.args.max_steps):
            result = self.infer_step(self.args.task)
            if result is None:
                time.sleep(0.002)
                continue
            if self.stop_requested:
                break
            self.publish_command(result.command_q23)
            action, elapsed_ms = result.action, result.elapsed_ms

            if step % max(1, self.args.print_every) == 0:
                rtc_text = (
                    f" vlm_ms={result.inference_ms:.1f}"
                    if result.inference_ms is not None
                    else ""
                )
                print(
                    f"step={step} loop_ms={elapsed_ms:.1f}{rtc_text} "
                    f"left_gripper={action[14]:.3f} right_gripper={action[15]:.3f}"
                )
            step += 1
            next_deadline += period_sec
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.monotonic()


class InteractiveDeploymentSession:
    """Keep one enabled PI0.5 policy alive while accepting terminal commands.

    The command reader runs separately from inference. ``SessionControl``
    invalidates any action that was inferred before a start, stop, continue, or
    exit transition, then authorizes publication immediately before ROS output.
    """

    def __init__(self, deployer: LocalPI05Deployer) -> None:
        self.deployer = deployer
        self.control = SessionControl()
        self._exit_requested = threading.Event()
        self._command_thread: threading.Thread | None = None

    def request_exit(self) -> None:
        self._exit_requested.set()
        self.control.request_exit()
        self.deployer.pause_task()
        self.deployer.stop_requested = True

    def _enable(self) -> None:
        if self.deployer.model_enabled:
            print("Model is already enabled.")
            return
        try:
            self.deployer.enable()
        except Exception as exc:
            print(f"enable failed: {exc}")
            return
        self.control.mark_enabled()

    def _start(self, task: str) -> None:
        task = task.strip()
        if not task:
            print("start failed: start requires task text, for example: start pick up the water bottle")
            return
        generation = self.control.begin_start()
        try:
            self.deployer.start_task(task)
        except (RuntimeError, ValueError) as exc:
            self.control.abort_transition(generation)
            print(f"start failed: {exc}")
            return
        if self.control.finish_start(generation, task):
            print(f"Started task: {task!r}")

    def _stop(self) -> None:
        if self.control.pause():
            self.deployer.pause_task()
            print("Inference paused. The model remains loaded; use continue to resume.")
        else:
            print("Inference is not running.")

    def _continue(self) -> None:
        pending = self.control.begin_continue(self.deployer.args.max_steps)
        if pending is None:
            print("No task is available to continue. Use start <task text>.")
            return
        task, generation = pending
        try:
            # Keep the task and loaded weights, but discard actions inferred
            # before the pause. The next step uses the current robot state.
            self.deployer.start_task(task)
        except (RuntimeError, ValueError) as exc:
            self.control.abort_transition(generation)
            print(f"continue failed: {exc}")
            return
        if not self.control.finish_continue(generation):
            return
        print(f"Continuing task: {task!r}")

    def _status(self) -> None:
        snapshot = self.control.snapshot()
        print(
            f"status: mode={snapshot.mode}, model_enabled={self.deployer.model_enabled}, "
            f"steps={snapshot.step}, task={snapshot.task!r}"
        )

    def handle_command(self, line: str) -> None:
        command, _, remainder = line.strip().partition(" ")
        command = command.lower()
        if not command:
            return
        if command == "enable":
            self._enable()
        elif command == "start":
            self._start(remainder)
        elif command == "stop":
            self._stop()
        elif command == "continue":
            self._continue()
        elif command == "status":
            self._status()
        elif command in {"help", "?"}:
            self.print_help()
        elif command == "exit":
            print("Exiting deployment session.")
            self.request_exit()
        else:
            print(f"Unknown command: {command!r}. Type help for available commands.")

    @staticmethod
    def print_help() -> None:
        print(
            "Commands:\n"
            "  enable              Load and warm up the model without publishing commands.\n"
            "  start <task text>   Reset PI0.5 and begin a new task.\n"
            "  stop                Pause publishing and retain the model in memory.\n"
            "  continue            Resume the paused task from a fresh observation.\n"
            "  status              Show the session state.\n"
            "  exit                Stop publishing, release resources, and exit."
        )

    def _read_commands(self) -> None:
        while not self._exit_requested.is_set():
            line = sys.stdin.readline()
            if not line:
                print("Command input closed; exiting deployment session.")
                self.request_exit()
                return
            self.handle_command(line)

    def run(self) -> None:
        self.print_help()
        self._command_thread = threading.Thread(target=self._read_commands, daemon=True)
        self._command_thread.start()
        period_sec = 1.0 / max(self.deployer.args.hz, 1e-6)
        next_deadline = time.monotonic()

        while not self._exit_requested.is_set() and not self.deployer.stop_requested:
            snapshot = self.control.snapshot()
            if snapshot.mode != "running":
                time.sleep(0.05)
                next_deadline = time.monotonic()
                continue

            result = self.deployer.infer_step(snapshot.task)
            if result is None:
                time.sleep(0.002)
                continue
            published = self.control.publish_if_current(
                snapshot.generation,
                lambda: self.deployer.publish_command(result.command_q23),
            )
            if not published:
                continue
            if snapshot.step % max(1, self.deployer.args.print_every) == 0:
                rtc_text = (
                    f" vlm_ms={result.inference_ms:.1f}"
                    if result.inference_ms is not None
                    else ""
                )
                print(
                    f"step={snapshot.step} loop_ms={result.elapsed_ms:.1f}{rtc_text} "
                    f"left_gripper={result.action[14]:.3f} right_gripper={result.action[15]:.3f}"
                )
            reached_limit = self.control.record_published_step(
                snapshot.generation,
                self.deployer.args.max_steps,
            )
            if reached_limit:
                self.deployer.pause_task()
                print("Maximum step count reached; inference paused.")

            next_deadline += period_sec
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.monotonic()


def parse_args() -> argparse.Namespace:
    model_dir_env = os.getenv("MODEL_DIR")
    require_model_dir = env_bool("PI05_REQUIRE_MODEL_DIR", False) and not model_dir_env
    repo_default = (
        None
        if require_model_dir
        else Path(model_dir_env or str(REPO_DIR / "004500" / "pretrained_model"))
    )
    script_name = Path(sys.argv[0]).name
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            f"  {script_name} --model-dir /home/ubuntu/model --task 'pick up the bottle'\n"
            f"  {script_name} --with-head --with-waist --with-depth --dry-run --max-steps 30\n"
            f"  {script_name} --with-head --help"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_default,
        required=require_model_dir,
        help="Local LeRobot model directory to load.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(os.getenv("PI05_TOKENIZER_DIR")) if os.getenv("PI05_TOKENIZER_DIR") else None,
        help=(
            "Optional local PaliGemma tokenizer directory. Use this when the checkpoint's tokenizer "
            "is not cached or the robot cannot access Hugging Face."
        ),
    )
    parser.add_argument("--task", default="mango_pick", help="Natural-language task prompt passed to the PI0.5 tokenizer.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the process alive and accept enable/start/stop/continue/exit commands from the terminal.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used for policy inference, such as cuda or cpu.")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "checkpoint",
            "disabled",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default=os.getenv("PI05_COMPILE_MODE", "checkpoint"),
        help=(
            "Override the checkpoint torch.compile setting. checkpoint preserves its config; "
            "disabled runs eager PyTorch."
        ),
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=int(os.environ["N_ACTION_STEPS"]) if os.getenv("N_ACTION_STEPS") else None,
        help=(
            "Number of actions executed from each chunk before non-RTC PI0.5 infers again. "
            "Omit to preserve the checkpoint value; RTC uses --rtc-refresh-steps instead."
        ),
    )
    parser.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=env_bool("PI05_RTC", False),
        help=(
            "Enable asynchronous Real-Time Chunking. The control loop consumes queued actions "
            "while PI0.5 refreshes a corrected chunk from the newest observation."
        ),
    )
    parser.add_argument(
        "--rtc-refresh-steps",
        type=int,
        default=int(os.environ["RTC_REFRESH_STEPS"]) if os.getenv("RTC_REFRESH_STEPS") else None,
        help=(
            "Start the next RTC VLM inference after this many actions from the latest chunk. "
            "Omit to derive 20 steps from LeRobot's 30-actions-remaining default for chunk_size=50."
        ),
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=int(os.getenv("RTC_EXECUTION_HORIZON", "10")),
        help="Number of leftover actions supplied as the RTC correction prefix.",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=float(os.getenv("RTC_MAX_GUIDANCE_WEIGHT", "10.0")),
        help="Maximum RTC prefix-guidance weight used during action denoising.",
    )
    parser.add_argument("--hz", type=float, default=10.0, help="Robot command publication and inference loop frequency in Hz.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum inference steps; 0 runs until Ctrl-C.")
    parser.add_argument("--warmup-sec", type=float, default=10.0, help="Maximum time to wait for a complete joint state and camera frame.")
    parser.add_argument("--start-delay-sec", type=float, default=3.0, help="Delay after readiness before commands are published, allowing the operator to cancel.")
    parser.add_argument(
        "--camera",
        dest="camera_names",
        action="append",
        choices=sorted(DEFAULT_RGB_CAMERA_NAMES),
        default=None,
        help="RGB SHM camera feature to provide. Repeat to select a subset; default is rgbd_head_color, hand_left, and hand_right.",
    )
    parser.add_argument(
        "--with-depth",
        action=argparse.BooleanOptionalAction,
        default=env_bool("WITH_DEPTH", False),
        help="Append the rgbd_head_depth observation.images feature and quantize live uint16 millimetre depth like the collector.",
    )
    parser.add_argument(
        "--depth-camera",
        dest="depth_camera_names",
        action="append",
        choices=sorted(DEFAULT_DEPTH_CAMERA_NAMES),
        default=None,
        help="Depth SHM camera feature to provide. Repeat for multiple depth cameras; implies --with-depth.",
    )
    parser.add_argument("--joints-topic", default=None, help="Override the complete physical q23 joint-state ROS2 topic.")
    parser.add_argument("--dry-run", action="store_true", help="Run inference without publishing arm, gripper, head, or waist commands.")
    parser.add_argument(
        "--with-head", "--control-head", dest="with_head",
        action=argparse.BooleanOptionalAction, default=env_bool("WITH_HEAD", False),
        help="Include neck_roll, neck_pitch, and neck_yaw in the policy state/action and publish their predicted commands.",
    )
    parser.add_argument(
        "--with-waist", "--control-waist", dest="with_waist",
        action=argparse.BooleanOptionalAction, default=env_bool("WITH_WAIST", False),
        help="Include leg_ankle, leg_knee, waist_pitch, and waist_yaw in the policy state/action and publish their predicted commands.",
    )
    parser.add_argument(
        "--sync-reference-camera",
        choices=sorted(CAMERA_SPECS),
        default=os.getenv("SYNC_REFERENCE_CAMERA", DEFAULT_SYNC_REFERENCE_CAMERA),
        help="Camera timestamp anchor for the multi-camera observation; it must be selected in the active camera list.",
    )
    parser.add_argument(
        "--max-image-delta-sec",
        type=float,
        default=float(os.getenv("MAX_IMAGE_DELTA_SEC", DEFAULT_MAX_IMAGE_DELTA_SEC)),
        help="Maximum timestamp difference between selected camera frames in one observation.",
    )
    parser.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_MIN, help="Minimum depth range in metres, matching collector encoding.")
    parser.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_MAX, help="Maximum depth range in metres, matching collector encoding.")
    parser.add_argument("--depth-shift", type=float, default=DEFAULT_DEPTH_SHIFT, help="Logarithmic depth shift in metres, matching collector encoding.")
    parser.add_argument(
        "--depth-use-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use logarithmic depth quantization, matching the collector default.",
    )
    parser.add_argument("--gripper-int", action=argparse.BooleanOptionalAction, default=True, help="Round gripper targets to integer values before publishing; use --no-gripper-int for floating-point values.")
    parser.add_argument("--max-state-age-sec", type=float, default=0.20, help="Reject inference when the newest joint state is older than this many seconds.")
    parser.add_argument("--max-image-age-sec", type=float, default=0.20, help="Reject inference when the newest camera frame is older than this many seconds.")
    parser.add_argument("--print-every", type=int, default=10, help="Print one inference timing/status line every N steps.")
    show_requested_parameter_help(parser)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    if args.tokenizer_dir is not None:
        args.tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    args.camera_names = list(dict.fromkeys(args.camera_names or DEFAULT_RGB_CAMERA_NAMES))
    args.depth_camera_names = list(dict.fromkeys(args.depth_camera_names or []))
    if args.with_depth and not args.depth_camera_names:
        args.depth_camera_names = list(DEFAULT_DEPTH_CAMERA_NAMES)
    if args.depth_camera_names:
        args.with_depth = True
    for name in args.depth_camera_names:
        if name in args.camera_names:
            raise SystemExit(f"Camera {name} cannot be both RGB and depth.")
    args.camera_names.extend(name for name in args.depth_camera_names if name not in args.camera_names)
    if args.sync_reference_camera not in args.camera_names:
        raise SystemExit(
            f"--sync-reference-camera {args.sync_reference_camera!r} is not active. "
            f"Active cameras: {', '.join(args.camera_names)}"
        )
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model_dir}")
    if (
        args.hz <= 0
        or (args.n_action_steps is not None and args.n_action_steps <= 0)
        or (args.rtc_refresh_steps is not None and args.rtc_refresh_steps <= 0)
        or args.rtc_execution_horizon <= 0
        or args.rtc_max_guidance_weight <= 0
        or args.max_steps < 0
        or args.warmup_sec <= 0
        or args.start_delay_sec < 0
        or args.print_every <= 0
        or args.max_state_age_sec <= 0
        or args.max_image_age_sec <= 0
        or args.max_image_delta_sec <= 0
        or args.depth_min < 0
        or args.depth_max <= args.depth_min
    ):
        raise SystemExit(
            "Action/RTC settings, hz, warmup, print interval, freshness limits, image delta, "
            "and depth range must be positive; max steps and start delay cannot be negative."
        )
    return args


def main() -> None:
    args = parse_args()
    deployer: LocalPI05Deployer | None = None
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
        deployer = LocalPI05Deployer(args)
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
