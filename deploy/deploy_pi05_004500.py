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
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
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
# reusing the camera ABI and canonical schema from the collector.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
COLLECTOR_DIR = REPO_DIR / "lerobot_data_collector"
for import_dir in (REPO_DIR, COLLECTOR_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from camera_config import CAMERA_SPECS, CameraSpec  # noqa: E402
from robot_schema import RobotSchema, build_robot_schema  # noqa: E402
from shm_camera import read_shm_frame, read_shm_metadata, shm_timestamp_sec  # noqa: E402


WHOLE_BODY_DIM = 23
BASE_POLICY_DIM = 16
HEAD_DIM = 3
WAIST_DIM = 4
MODEL_IMAGE_KEY = "observation.images.rgbd_head_color"
MODEL_IMAGE_SHAPE = (3, 480, 640)


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


def expected_policy_dim(with_head: bool, with_waist: bool) -> int:
    """Return the collector schema size in canonical policy order."""

    return BASE_POLICY_DIM + (HEAD_DIM if with_head else 0) + (WAIST_DIM if with_waist else 0)


def flatten_joint_positions(packet: dict[str, Any]) -> np.ndarray | None:
    """Read the fixed physical q23 order from a robot status JSON packet.

    Missing or short groups are rejected rather than silently padded. Padding
    a live robot state could make a policy issue a large, unexpected command.
    """

    groups = (
        ("leg_waist_joint_state", 4),
        ("left_arm_joint_state", 7),
        ("right_arm_joint_state", 7),
        ("left_gripper_state", 1),
        ("right_gripper_state", 1),
        ("neck_joint_state", 3),
    )
    values: list[float] = []
    for key, size in groups:
        group = packet.get(key)
        positions = group.get("position") if isinstance(group, dict) else None
        if not isinstance(positions, (list, tuple)) or len(positions) < size:
            return None
        try:
            values.extend(float(value) for value in positions[:size])
        except (TypeError, ValueError):
            return None
    return np.asarray(values, dtype=np.float32)


def policy_state_from_q23(q23: np.ndarray, schema: RobotSchema) -> np.ndarray:
    """Convert physical q23 into the collector's canonical state order."""

    q = np.asarray(q23, dtype=np.float32).reshape(-1)
    if q.size < WHOLE_BODY_DIM:
        raise ValueError(f"Expected {WHOLE_BODY_DIM} physical joints, got {q.size}.")

    values = [*q[4:11], *q[11:18], q[18], q[19]]
    if schema.groups[-1].name == "head" or any(group.name == "head" for group in schema.groups):
        values.extend(q[20:23])
    if any(group.name == "waist" for group in schema.groups):
        values.extend(q[:4])
    state = np.asarray(values, dtype=np.float32)
    if state.size != schema.size:
        raise AssertionError(f"Internal schema conversion produced {state.size}, expected {schema.size}.")
    return state


def physical_q23_from_action(action: np.ndarray, latest_q23: np.ndarray, schema: RobotSchema) -> np.ndarray:
    """Map canonical policy action back to the physical ROS q23 order."""

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < schema.size:
        raise ValueError(f"Policy returned {action.size} dims, expected {schema.size}.")
    q = np.asarray(latest_q23, dtype=np.float32).reshape(-1)[:WHOLE_BODY_DIM].copy()

    # Base dimensions are always controlled by this policy.
    q[4:11] = action[0:7]
    q[11:18] = action[7:14]
    q[18] = action[14]
    q[19] = action[15]

    offset = BASE_POLICY_DIM
    if any(group.name == "head" for group in schema.groups):
        q[20:23] = action[offset : offset + HEAD_DIM]
        offset += HEAD_DIM
    if any(group.name == "waist" for group in schema.groups):
        q[:4] = action[offset : offset + WAIST_DIM]
    return q


def image_to_policy_chw(image: np.ndarray) -> np.ndarray:
    """Convert a BGR SHM image into the PI0.5 RGB uint8 CHW contract."""

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
    image = cv2.resize(image, (MODEL_IMAGE_SHAPE[2], MODEL_IMAGE_SHAPE[1]), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    if chw.shape != MODEL_IMAGE_SHAPE:
        raise ValueError(f"Image has shape {chw.shape}, expected {MODEL_IMAGE_SHAPE}.")
    return chw


def load_policy(model_dir: Path, device: str):
    """Load PI0.5 with the public LeRobot API used by the 004500 checkpoint."""

    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError as exc:
        raise SystemExit(
            "Cannot import PI05Policy. Activate the LeRobot environment used for training."
        ) from exc

    policy = PI05Policy.from_pretrained(str(model_dir))
    policy = policy.to(device).eval()
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


def read_model_contract(model_dir: Path) -> tuple[int | None, int | None, tuple[int, ...] | None]:
    """Read dimensions from config.json before touching the robot controller."""

    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None, None, None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        input_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
        action_shape = config.get("output_features", {}).get("action", {}).get("shape")
        image_shape = config.get("input_features", {}).get(MODEL_IMAGE_KEY, {}).get("shape")
        state_dim = int(input_shape[0]) if isinstance(input_shape, list) and input_shape else None
        action_dim = int(action_shape[0]) if isinstance(action_shape, list) and action_shape else None
        parsed_image_shape = tuple(int(value) for value in image_shape) if image_shape else None
        return state_dim, action_dim, parsed_image_shape
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse model contract: {config_path}: {exc}") from exc


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
        q23 = flatten_joint_positions(packet)
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


class DirectShmCamera:
    """Read only new, internally consistent frames from one camera's SHM ABI."""

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self.last_timestamp_ns: int | None = None
        self.latest_received_sec = 0.0

    def read_latest(self) -> np.ndarray | None:
        metadata = read_shm_metadata(self.spec)
        if metadata is None:
            return None
        if self.last_timestamp_ns == metadata[0]:
            return None
        frame = read_shm_frame(self.spec, metadata)
        if frame is None or frame.timestamp_ns != metadata[0]:
            return None
        if frame.pixel_format != 1 or frame.channels != 3:
            raise ValueError(
                f"{self.spec.name} must be pixel_format=1/channels=3, "
                f"got {frame.pixel_format}/{frame.channels}"
            )
        bgr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        self.last_timestamp_ns = frame.timestamp_ns
        self.latest_received_sec = time.time()
        # The timestamp is intentionally evaluated here as a diagnostic hook:
        # current robot SHM timestamps may be device/monotonic values.
        shm_timestamp_sec(frame.timestamp_ns, self.latest_received_sec)
        return np.ascontiguousarray(bgr)


class LocalPI05Deployer:
    """Coordinate state/image snapshots, policy inference, and robot commands."""

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

        contract_state, contract_action, contract_image = read_model_contract(args.model_dir)
        expected_dim = self.schema.size
        if contract_state is not None and contract_state != expected_dim:
            raise ValueError(
                f"Model expects observation.state={contract_state} dims, but the selected "
                f"head/waist switches produce {expected_dim}. Use the switches used during training."
            )
        if contract_action is not None and contract_action < expected_dim:
            raise ValueError(
                f"Model action has {contract_action} dims, but the selected schema needs {expected_dim}."
            )
        if contract_image is not None and contract_image != MODEL_IMAGE_SHAPE:
            raise ValueError(
                f"Model image shape is {contract_image}, but this deployer supplies {MODEL_IMAGE_SHAPE}."
            )

        rclpy.init(args=None)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.telemetry = JointTelemetryNode(self.joints_topic)
        self.command_node = AutoLifeCommandNode()
        self.executor.add_node(self.telemetry)
        self.executor.add_node(self.command_node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        camera_name = args.camera
        self.camera = DirectShmCamera(CAMERA_SPECS[camera_name])
        self.latest_image_chw: np.ndarray | None = None
        self.latest_image_received_sec = 0.0
        self.policy = load_policy(args.model_dir, args.device)

        print(f"policy schema: {', '.join(self.schema.names)}")
        print(f"policy state/action dimensions: {expected_dim}/{contract_action or 'unknown'}")
        print(f"camera source: SHM {camera_name} ({CAMERA_SPECS[camera_name].meta_path})")

    def close(self) -> None:
        self.stop_requested = True
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

    def refresh(self) -> tuple[np.ndarray | None, float]:
        assert self.telemetry is not None
        q23, state_received_sec = self.telemetry.snapshot()
        image = self.camera.read_latest()
        if image is not None:
            self.latest_image_chw = image_to_policy_chw(image)
            self.latest_image_received_sec = self.camera.latest_received_sec
        return q23, state_received_sec

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.args.warmup_sec
        while not self.stop_requested and time.monotonic() < deadline:
            q23, state_received_sec = self.refresh()
            image_ready = self.latest_image_chw is not None
            state_ready = q23 is not None
            if state_ready and image_ready:
                print("Ready: joint state and camera frame are available.")
                return
            time.sleep(0.01)
        missing = []
        if self.telemetry is None or self.telemetry.latest_q23 is None:
            missing.append(f"joint state on {self.joints_topic}")
        if self.latest_image_chw is None:
            missing.append(f"camera {self.args.camera}")
        raise TimeoutError(f"Timed out waiting for: {', '.join(missing)}")

    def build_observation(self, q23: np.ndarray) -> dict[str, Any]:
        if self.latest_image_chw is None:
            raise RuntimeError("Camera observation is not ready.")
        state = policy_state_from_q23(q23, self.schema)
        return {
            "observation.state": torch.from_numpy(state),
            MODEL_IMAGE_KEY: torch.from_numpy(self.latest_image_chw.astype(np.float32) / 255.0),
            "task": self.args.task,
        }

    def select_action(self, q23: np.ndarray) -> np.ndarray:
        assert self.policy is not None
        observation = self.build_observation(q23)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
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
        self.wait_until_ready()
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
            loop_started = time.monotonic()
            q23, state_received_sec = self.refresh()
            now = time.time()
            if q23 is None or self.latest_image_chw is None:
                time.sleep(0.002)
                continue
            if now - state_received_sec > self.args.max_state_age_sec:
                time.sleep(0.002)
                continue
            if now - self.latest_image_received_sec > self.args.max_image_age_sec:
                time.sleep(0.002)
                continue

            action = self.select_action(q23)
            command_q23 = physical_q23_from_action(action, q23, self.schema)
            if not self.args.dry_run:
                assert self.command_node is not None
                self.command_node.publish_action(command_q23, self.args.gripper_int)

            if step % max(1, self.args.print_every) == 0:
                elapsed_ms = (time.monotonic() - loop_started) * 1000.0
                print(
                    f"step={step} loop_ms={elapsed_ms:.1f} "
                    f"left_gripper={action[14]:.3f} right_gripper={action[15]:.3f}"
                )
            step += 1
            next_deadline += period_sec
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.monotonic()


def parse_args() -> argparse.Namespace:
    repo_default = REPO_DIR / "004500" / "pretrained_model"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=repo_default)
    parser.add_argument("--task", default="mango_pick")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Ctrl-C")
    parser.add_argument("--warmup-sec", type=float, default=10.0)
    parser.add_argument("--start-delay-sec", type=float, default=3.0)
    parser.add_argument("--camera", choices=sorted(CAMERA_SPECS), default="rgbd_head_color")
    parser.add_argument("--joints-topic", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Infer without publishing commands")
    parser.add_argument(
        "--with-head", "--control-head", dest="with_head",
        action=argparse.BooleanOptionalAction, default=env_bool("WITH_HEAD", False),
        help="Include neck joints in policy state/action and control them.",
    )
    parser.add_argument(
        "--with-waist", "--control-waist", dest="with_waist",
        action=argparse.BooleanOptionalAction, default=env_bool("WITH_WAIST", False),
        help="Include leg/waist joints in policy state/action and control them.",
    )
    parser.add_argument("--gripper-int", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-state-age-sec", type=float, default=0.20)
    parser.add_argument("--max-image-age-sec", type=float, default=0.20)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model_dir}")
    if args.hz <= 0 or args.max_state_age_sec <= 0 or args.max_image_age_sec <= 0:
        raise SystemExit("--hz and freshness limits must be positive.")
    return args


def main() -> None:
    args = parse_args()
    deployer: LocalPI05Deployer | None = None

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        if deployer is not None:
            deployer.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        deployer = LocalPI05Deployer(args)
        deployer.run()
    except KeyboardInterrupt:
        pass
    finally:
        if deployer is not None:
            deployer.close()


if __name__ == "__main__":
    main()
