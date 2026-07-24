#!/usr/bin/env python3
"""Mirror one real AutoLife robot in Isaac Sim from its ROS 2 joint status."""

from __future__ import annotations

import argparse
import base64
import fcntl
import html
import ipaddress
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from joint_mapping import PHYSICAL_JOINT_NAMES, JointSample, parse_status_json, with_mimic_positions


DEFAULT_URDF = Path(
    "/home/wayne-cb/Desktop/autolife_s1(1)/autolife_s1/urdfs/robot_v2_2.urdf"
)

RMW_LIBRARY_NAMES = {
    "rmw_cyclonedds_cpp": "librmw_cyclonedds_cpp.so",
    "rmw_fastrtps_cpp": "librmw_fastrtps_cpp.so",
}

KNOWN_ROBOT_HOSTS = {
    "283": "192.168.8.42",
    "300": "192.168.8.202",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", required=True, help="Robot topic suffix, for example 283.")
    parser.add_argument(
        "--robot-host",
        default=None,
        help=(
            "Robot SSH host/IP. Known IDs 283 and 300 are resolved automatically; "
            "set this explicitly if an address changes."
        ),
    )
    parser.add_argument("--robot-user", default="ubuntu", help="SSH user for remote telemetry.")
    parser.add_argument(
        "--transport",
        choices=("auto", "ros2", "ssh"),
        default="auto",
        help="Telemetry transport. Auto uses SSH for known/configured remote robots.",
    )
    parser.add_argument("--ros-domain-id", type=int, default=0, help="ROS_DOMAIN_ID used by the robot.")
    parser.add_argument(
        "--state-topic",
        default=None,
        help="Override the complete status topic; normally derived from domain and robot ID.",
    )
    parser.add_argument(
        "--network-interface-address",
        default=None,
        help="Optional CycloneDDS interface name/IP. Omit to let DDS select a LAN interface.",
    )
    parser.add_argument(
        "--rmw-implementation",
        default="auto",
        help=(
            "ROS 2 RMW implementation. 'auto' prefers CycloneDDS and falls back "
            "to Fast DDS when CycloneDDS is not installed."
        ),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--usd-output-dir",
        type=Path,
        default=Path("/tmp/autolife_digital_twin"),
        help="Directory for generated USD assets.",
    )
    parser.add_argument(
        "--source",
        choices=("measured", "target"),
        default="measured",
        help="Mirror measured joints or targets embedded in the status packet.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without a window.")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-ground", action="store_true", help="Do not create a ground plane.")
    parser.add_argument("--ground-z", type=float, default=0.0, help="Ground plane height in metres.")
    parser.add_argument("--ground-size", type=float, default=20.0, help="Ground plane edge length in metres.")
    parser.add_argument(
        "--ground-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(0.12, 0.32, 0.48),
        help="Ground RGB colour, with each component between 0 and 1.",
    )
    parser.add_argument(
        "--initial-state-timeout-sec",
        type=float,
        default=0.0,
        help="Exit if no state arrives within this time; 0 waits indefinitely (default).",
    )
    parser.add_argument(
        "--stale-warning-sec",
        type=float,
        default=1.0,
        help="Warn when telemetry is this old; the twin freezes until data resumes.",
    )
    parser.add_argument(
        "--print-period-sec", type=float, default=5.0, help="Periodic status log interval."
    )
    return parser.parse_args()


def _library_search_dirs() -> list[Path]:
    directories: list[Path] = []
    for value in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep):
        if value:
            directories.append(Path(value))
    for prefix in os.getenv("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix:
            directories.append(Path(prefix) / "lib")
    ros_distro = os.getenv("ROS_DISTRO")
    if ros_distro:
        directories.append(Path("/opt/ros") / ros_distro / "lib")
    else:
        directories.extend(sorted(Path("/opt/ros").glob("*/lib")))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(directories))


def rmw_is_installed(identifier: str) -> bool:
    library_name = RMW_LIBRARY_NAMES.get(identifier, f"lib{identifier}.so")
    return any((directory / library_name).is_file() for directory in _library_search_dirs())


def resolve_rmw_implementation(requested: str) -> str:
    if requested != "auto":
        if rmw_is_installed(requested):
            return requested
        hint = (
            " Install it with 'sudo apt install ros-jazzy-rmw-cyclonedds-cpp', "
            "or use '--rmw-implementation auto'."
            if requested == "rmw_cyclonedds_cpp"
            else ""
        )
        raise RuntimeError(
            f"Requested ROS 2 RMW implementation {requested!r} is not installed.{hint}"
        )
    for identifier in ("rmw_cyclonedds_cpp", "rmw_fastrtps_cpp"):
        if rmw_is_installed(identifier):
            return identifier
    raise RuntimeError(
        "No supported ROS 2 RMW implementation was found. Install either "
        "rmw_cyclonedds_cpp or rmw_fastrtps_cpp."
    )


def cyclonedds_interface_uri(interface: str) -> str:
    """Build a current CycloneDDS interface selector for an IP or device name."""

    try:
        ipaddress.ip_address(interface)
        selector = f'address="{html.escape(interface, quote=True)}"'
    except ValueError:
        selector = f'name="{html.escape(interface, quote=True)}"'
    return (
        "<CycloneDDS><Domain><General><Interfaces>"
        f'<NetworkInterface {selector} multicast="false"/>'
        "</Interfaces></General></Domain></CycloneDDS>"
    )


def configure_ros(args: argparse.Namespace) -> tuple[str, str]:
    rmw_implementation = resolve_rmw_implementation(args.rmw_implementation)
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    os.environ["RMW_IMPLEMENTATION"] = rmw_implementation
    if args.network_interface_address:
        if rmw_implementation != "rmw_cyclonedds_cpp":
            raise RuntimeError(
                "--network-interface-address configures CycloneDDS, but the selected "
                f"implementation is {rmw_implementation}. Install CycloneDDS or omit "
                "the interface option to use automatic Fast DDS discovery."
            )
        os.environ["CYCLONEDDS_URI"] = cyclonedds_interface_uri(
            args.network_interface_address
        )
    topic = args.state_topic or (
        "/topic_arm_whole_body_and_gripper_current_joints_status_"
        f"{args.ros_domain_id}_{args.robot_id}"
    )
    return topic, rmw_implementation


def inspect_urdf(path: Path) -> tuple[dict[str, tuple[str, float, float]], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {path}")
    root = ET.parse(path).getroot()
    joints: set[str] = set()
    mimic_rules: dict[str, tuple[str, float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        joints.add(name)
        mimic = joint.find("mimic")
        if mimic is not None and mimic.get("joint"):
            mimic_rules[name] = (
                str(mimic.get("joint")),
                float(mimic.get("multiplier", "1")),
                float(mimic.get("offset", "0")),
            )
    missing = sorted(set(PHYSICAL_JOINT_NAMES) - joints)
    if missing:
        raise ValueError(f"URDF is missing required joints: {', '.join(missing)}")
    return mimic_rules, joints


class LatestJointState:
    def __init__(self, source: str) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._sample: JointSample | None = None
        self._received_monotonic = 0.0
        self._sequence = 0
        self.invalid_packets = 0

    def update(self, text: str) -> None:
        sample = parse_status_json(text, source=self.source)
        if sample is None:
            self.invalid_packets += 1
            return
        with self._lock:
            self._sample = sample
            self._received_monotonic = time.monotonic()
            self._sequence += 1

    def snapshot(self) -> tuple[JointSample | None, float, int]:
        with self._lock:
            return self._sample, self._received_monotonic, self._sequence


class SSHJointStream:
    """Forward a loopback-only ROS topic from a robot over SSH."""

    _IGNORED_CYCLONEDDS_MESSAGES = (
        'selected interface "lo" is not multicast-capable: disabling multicast',
    )

    _REMOTE_READER = """
import base64
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

rclpy.init()
node = Node("autolife_digital_twin_forwarder")
node.create_subscription(
    String,
    sys.argv[1],
    lambda message: print(
        base64.b64encode(message.data.encode("utf-8")).decode("ascii"),
        flush=True,
    ),
    20,
)
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
"""

    def __init__(
        self,
        host: str,
        user: str,
        topic: str,
        ros_domain_id: int,
        latest: LatestJointState,
    ) -> None:
        remote_command = " ".join(
            (
                "source /opt/ros/jazzy/setup.bash;",
                f"export ROS_DOMAIN_ID={int(ros_domain_id)};",
                "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp;",
                "export CYCLONEDDS_URI="
                + shlex.quote(cyclonedds_interface_uri("lo"))
                + ";",
                "/usr/bin/python3 -u -c",
                shlex.quote(self._REMOTE_READER),
                shlex.quote(topic),
            )
        )
        self.process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", f"{user}@{host}", remote_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.latest = latest
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                text = base64.b64decode(line.strip(), validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            self.latest.update(text)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            text = line.rstrip()
            if text and not any(
                message in text for message in self._IGNORED_CYCLONEDDS_MESSAGES
            ):
                print(f"[digital_twin][ssh] {text}", file=sys.stderr)

    def poll(self) -> int | None:
        return self.process.poll()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def find_articulation_root(stage) -> str:
    from pxr import UsdPhysics

    roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not roots:
        raise RuntimeError("The imported USD stage contains no articulation root.")
    roots.sort(key=lambda path: (path.count("/"), len(path)))
    return roots[0]


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.ground_size, args.stale_warning_sec, args.print_period_sec) <= 0:
        raise SystemExit("Resolution, ground size, and warning/period values must be positive.")
    if args.initial_state_timeout_sec < 0:
        raise SystemExit("--initial-state-timeout-sec must be zero or positive.")
    if any(component < 0.0 or component > 1.0 for component in args.ground_color):
        raise SystemExit("--ground-color components must be between 0 and 1.")

    lock_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(args.robot_id))
    lock_stream = open(f"/tmp/autolife_digital_twin_{lock_id}.lock", "w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"A digital twin for robot {args.robot_id} is already running. "
            "Close the existing Isaac Sim window before starting another one."
        )

    topic, rmw_implementation = configure_ros(args)
    mimic_rules, _ = inspect_urdf(args.urdf.resolve())
    args.usd_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[digital_twin] ROS RMW: {rmw_implementation}")

    # Isaac modules must be imported only after SimulationApp is constructed.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
        }
    )

    node = None
    ssh_stream = None
    simulation_context = None
    try:
        import numpy as np
        import omni.usd
        import rclpy
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import UsdLux
        from rclpy.node import Node
        from std_msgs.msg import String

        config = URDFImporterConfig(
            urdf_path=str(args.urdf.resolve()),
            usd_path=str(args.usd_output_dir.resolve()),
            merge_fixed_joints=False,
            merge_mesh=False,
            collision_from_visuals=False,
            fix_base=True,
            joint_drive_type="force",
            joint_target_type="position",
        )
        usd_path = URDFImporter(config).import_urdf()
        if not omni.usd.get_context().open_stage(usd_path):
            raise RuntimeError(f"Could not open imported USD stage: {usd_path}")
        simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        root_path = find_articulation_root(stage)
        light = UsdLux.DistantLight.Define(stage, "/World/DigitalTwinLight")
        light.CreateIntensityAttr(1000.0)
        if not args.no_ground:
            GroundPlane(
                prim_path="/World/GroundPlane",
                size=args.ground_size,
                z_position=args.ground_z,
                color=np.asarray(args.ground_color, dtype=np.float32),
            )
        if not args.headless:
            set_camera_view(
                eye=[3.0, 3.0, 2.0],
                target=[0.0, 0.0, 1.0],
                camera_prim_path="/OmniverseKit_Persp",
            )
        simulation_context = SimulationContext(stage_units_in_meters=1.0)
        simulation_context.initialize_physics()
        # The twin visualizes measured state; it must not be driven by gravity.
        # A normal (non-zero) physics step is still required for PhysX tensor
        # joint writes to reach the USD/viewport transforms reliably.
        simulation_context.get_physics_context().set_gravity(0.0)
        articulation = Articulation(root_path)
        articulation.initialize()
        simulation_context.play()
        simulation_context.step(render=not args.headless)
        simulation_context.set_simulation_dt(
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )

        dof_names = articulation.dof_names
        if not dof_names:
            raise RuntimeError("The imported articulation exposes no DOF names.")
        controllable_names = set(PHYSICAL_JOINT_NAMES) | set(mimic_rules)
        names = [name for name in dof_names if name in controllable_names]
        missing = sorted(set(PHYSICAL_JOINT_NAMES) - set(names))
        if missing:
            raise RuntimeError(
                "Imported articulation is missing mapped DOFs: " + ", ".join(missing)
            )
        indices = [articulation.get_dof_index(name) for name in names]
        zero_positions = np.zeros((1, len(names)), dtype=np.float32)
        articulation.set_joint_positions(zero_positions, joint_indices=indices)
        articulation.set_joint_velocities(zero_positions, joint_indices=indices)
        articulation.set_joint_position_targets(zero_positions, joint_indices=indices)

        latest = LatestJointState(args.source)
        robot_host = args.robot_host or KNOWN_ROBOT_HOSTS.get(str(args.robot_id))
        transport = args.transport
        if transport == "auto":
            transport = "ssh" if robot_host else "ros2"
        if transport == "ssh":
            if not robot_host:
                raise ValueError("--transport ssh requires --robot-host.")
            ssh_stream = SSHJointStream(
                robot_host,
                args.robot_user,
                topic,
                args.ros_domain_id,
                latest,
            )
        else:
            rclpy.init()
            node = Node(f"autolife_digital_twin_{args.robot_id}")
            node.create_subscription(String, topic, lambda message: latest.update(message.data), 20)

        print(f"[digital_twin] robot ID: {args.robot_id}")
        print(
            f"[digital_twin] transport: {transport}"
            + (f" ({args.robot_user}@{robot_host})" if transport == "ssh" else "")
        )
        print(f"[digital_twin] ROS topic: {topic}")
        print(f"[digital_twin] URDF: {args.urdf.resolve()}")
        print(f"[digital_twin] USD: {usd_path}")
        print(f"[digital_twin] articulation: {root_path} ({len(names)} controlled DOFs)")
        print(
            "[digital_twin] ground: "
            + (
                "disabled"
                if args.no_ground
                else (
                    f"z={args.ground_z:.3f} m, size={args.ground_size:.1f} m, "
                    f"rgb={tuple(args.ground_color)}"
                )
            )
        )
        print("[digital_twin] waiting for a complete q23 state ...")

        started = time.monotonic()
        last_sequence = 0
        last_log = started
        stale_reported = False
        last_applied_positions = None
        latest_max_change_deg = 0.0
        latest_pose_error_deg = 0.0
        while simulation_app.is_running():
            if node is not None:
                rclpy.spin_once(node, timeout_sec=0.0)
            if ssh_stream is not None and ssh_stream.poll() is not None:
                raise RuntimeError(
                    f"SSH telemetry process exited with code {ssh_stream.poll()}."
                )
            sample, received_at, sequence = latest.snapshot()
            now = time.monotonic()
            if (
                sample is None
                and args.initial_state_timeout_sec > 0
                and now - started > args.initial_state_timeout_sec
            ):
                raise TimeoutError(
                    f"No complete state received on {topic} within "
                    f"{args.initial_state_timeout_sec:.1f}s."
                )
            if sample is not None and sequence != last_sequence:
                expanded = with_mimic_positions(sample.urdf_positions_rad(), mimic_rules)
                positions = np.asarray([[expanded[name] for name in names]], dtype=np.float32)
                if last_applied_positions is not None:
                    latest_max_change_deg = float(
                        np.rad2deg(np.max(np.abs(positions - last_applied_positions)))
                    )
                last_applied_positions = positions
                last_sequence = sequence
                stale_reported = False
            if sample is not None and now - received_at > args.stale_warning_sec and not stale_reported:
                print(
                    f"[digital_twin] warning: telemetry is {now - received_at:.2f}s old; "
                    "freezing the twin until it resumes.",
                    file=sys.stderr,
                )
                stale_reported = True
            if now - last_log >= args.print_period_sec:
                if sample is None:
                    print(
                        f"[digital_twin] still waiting on {topic}; "
                        f"invalid={latest.invalid_packets}"
                    )
                else:
                    print(
                        f"[digital_twin] mirrored {sequence} packets; "
                        f"age={(now - received_at) * 1000.0:.1f} ms; "
                        f"last_max_change={latest_max_change_deg:.3f} deg; "
                        f"isaac_error={latest_pose_error_deg:.4f} deg; "
                        f"invalid={latest.invalid_packets}"
                    )
                last_log = now
            # Advance PhysX first, then overwrite the resulting dynamic state
            # immediately before rendering. The source URDF has no joint
            # stiffness/damping, so even with gravity disabled it can drift
            # during a physics step. Reapplying every display frame makes the
            # visible twin an exact state mirror.
            if not articulation.is_physics_handle_valid():
                break
            simulation_context.step(render=False)
            if (
                not simulation_app.is_running()
                or not articulation.is_physics_handle_valid()
            ):
                break
            if last_applied_positions is not None:
                articulation.set_joint_positions(
                    last_applied_positions, joint_indices=indices
                )
                articulation.set_joint_velocities(
                    np.zeros_like(last_applied_positions), joint_indices=indices
                )
                articulation.set_joint_position_targets(
                    last_applied_positions, joint_indices=indices
                )
            simulation_context.render()
            if (
                not simulation_app.is_running()
                or not articulation.is_physics_handle_valid()
            ):
                break
            if last_applied_positions is not None:
                actual_positions_raw = articulation.get_joint_positions(
                    joint_indices=indices
                )
                if actual_positions_raw is None:
                    break
                actual_positions = np.asarray(
                    actual_positions_raw, dtype=np.float32
                ).reshape(last_applied_positions.shape)
                latest_pose_error_deg = float(
                    np.rad2deg(
                        np.max(np.abs(actual_positions - last_applied_positions))
                    )
                )
    except Exception as exc:
        print(f"[digital_twin] fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        if ssh_stream is not None:
            ssh_stream.close()
        if node is not None:
            import rclpy

            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        if simulation_context is not None:
            simulation_context.stop()
        simulation_app.close()
        lock_stream.close()


if __name__ == "__main__":
    main()
