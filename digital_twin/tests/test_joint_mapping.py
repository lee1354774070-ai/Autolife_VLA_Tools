from __future__ import annotations

import math
import unittest
from pathlib import Path

from digital_twin.joint_mapping import (
    PHYSICAL_JOINT_NAMES,
    parse_status_payload,
    with_mimic_positions,
)
from digital_twin.mirror_robot import (
    cyclonedds_interface_uri,
    inspect_urdf,
    resolve_rmw_implementation,
    rmw_is_installed,
)


def status_packet() -> dict:
    return {
        "leg_waist_joint_state": {"position": [1, 2, 3, 4]},
        "left_arm_joint_state": {"position": list(range(5, 12))},
        "right_arm_joint_state": {"position": list(range(12, 19))},
        "left_gripper_state": {"position": [19]},
        "right_gripper_state": {"position": [20]},
        "neck_joint_state": {"position": [21, 22, 23]},
        "leg_waist_target_joint_state": [101, 102, 103, 104],
        "left_arm_target_joint_state": list(range(105, 112)),
        "right_arm_target_joint_state": list(range(112, 119)),
        "neck_target_joint_state": [121, 122, 123],
    }


class JointMappingTest(unittest.TestCase):
    def test_measured_packet_maps_q23_to_urdf_names(self) -> None:
        sample = parse_status_payload(status_packet())
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(len(PHYSICAL_JOINT_NAMES), 23)
        self.assertEqual(sample.positions_deg, tuple(range(1, 24)))
        positions = sample.urdf_positions_rad()
        self.assertAlmostEqual(positions["Joint_Ankle"], math.radians(1))
        self.assertAlmostEqual(positions["Joint_Neck_Yaw"], math.radians(23))

    def test_target_mode_keeps_measured_grippers(self) -> None:
        sample = parse_status_payload(status_packet(), source="target")
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(
            sample.positions_deg,
            tuple(range(101, 119)) + (19.0, 20.0, 121.0, 122.0, 123.0),
        )

    def test_partial_and_nonfinite_packets_are_rejected(self) -> None:
        packet = status_packet()
        packet["left_arm_joint_state"] = {"position": [1, 2]}
        self.assertIsNone(parse_status_payload(packet))
        packet = status_packet()
        packet["neck_joint_state"] = {"position": [1, float("nan"), 3]}
        self.assertIsNone(parse_status_payload(packet))

    def test_mimic_chain(self) -> None:
        result = with_mimic_positions(
            {"driver": 0.2},
            {"first": ("driver", -1.0, 0.1), "second": ("first", 2.0, 0.0)},
        )
        self.assertAlmostEqual(result["first"], -0.1)
        self.assertAlmostEqual(result["second"], -0.2)

    def test_configured_urdf_contains_all_required_joints(self) -> None:
        urdf = Path(
            "/home/wayne-cb/Desktop/autolife_s1(1)/autolife_s1/urdfs/robot_v2_2.urdf"
        )
        if not urdf.exists():
            self.skipTest(f"URDF is not available: {urdf}")
        mimic_rules, joints = inspect_urdf(urdf)
        self.assertTrue(set(PHYSICAL_JOINT_NAMES).issubset(joints))
        self.assertEqual(len(mimic_rules), 10)

    def test_rmw_auto_selects_an_installed_implementation(self) -> None:
        selected = resolve_rmw_implementation("auto")
        self.assertTrue(rmw_is_installed(selected))

    def test_cyclonedds_interface_uri_uses_current_schema(self) -> None:
        by_name = cyclonedds_interface_uri("lo")
        by_address = cyclonedds_interface_uri("127.0.0.1")
        self.assertIn('name="lo"', by_name)
        self.assertIn('address="127.0.0.1"', by_address)
        self.assertIn('multicast="false"', by_name)
        self.assertNotIn("NetworkInterfaceAddress", by_name)


if __name__ == "__main__":
    unittest.main()
