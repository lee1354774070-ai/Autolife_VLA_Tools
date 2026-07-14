#!/usr/bin/env python3
"""Unit tests for state/action ordering without requiring ROS or LeRobot."""

from __future__ import annotations

import unittest

import numpy as np

from robot_schema import build_robot_schema, parse_gripper_command


def status_payload() -> dict:
    """Build distinguishable values so an ordering error cannot hide."""

    return {
        "left_arm_joint_state": {"position": list(range(1, 8))},
        "right_arm_joint_state": {"position": list(range(11, 18))},
        "left_gripper_state": {"position": [21]},
        "right_gripper_state": {"position": [22]},
        "neck_joint_state": {"position": [31, 32, 33]},
        "leg_waist_joint_state": {"position": [41, 42, 43, 44]},
        "left_arm_target_joint_state": list(range(101, 108)),
        "right_arm_target_joint_state": list(range(111, 118)),
        "neck_target_joint_state": [131, 132, 133],
        "leg_waist_target_joint_state": [141, 142, 143, 144],
    }


class RobotSchemaTest(unittest.TestCase):
    def test_default_schema_preserves_16_dimension_prefix(self) -> None:
        schema = build_robot_schema()
        self.assertEqual(schema.size, 16)
        self.assertEqual(schema.names[-2:], ("left_gripper", "right_gripper"))
        np.testing.assert_array_equal(
            schema.parse_state(status_payload()),
            np.array([*range(1, 8), *range(11, 18), 21, 22], dtype=np.float32),
        )

    def test_optional_groups_are_appended_in_controller_order(self) -> None:
        schema = build_robot_schema(with_head=True, with_waist=True)
        self.assertEqual(schema.size, 23)
        self.assertEqual(
            schema.names[-7:],
            ("neck_roll", "neck_pitch", "neck_yaw", "leg_ankle", "leg_knee", "waist_pitch", "waist_yaw"),
        )
        np.testing.assert_array_equal(
            schema.parse_status_target(status_payload()),
            np.array(
                [*range(101, 108), *range(111, 118), 21, 22, 131, 132, 133, 141, 142, 143, 144],
                dtype=np.float32,
            ),
        )

    def test_joint_command_uses_the_same_dimension_order(self) -> None:
        schema = build_robot_schema(with_head=True, with_waist=True)
        body = schema.parse_body_command(
            {
                "left_arm_target_joints_position": list(range(101, 108)),
                "right_arm_target_joints_position": list(range(111, 118)),
                "neck_target_joints_position": [131, 132, 133],
                "leg_waist_target_joints_position": [141, 142, 143, 144],
            }
        )
        self.assertIsNotNone(body)
        action = schema.compose_command_action(body, (121, 122))
        np.testing.assert_array_equal(
            action,
            np.array(
                [*range(101, 108), *range(111, 118), 121, 122, 131, 132, 133, 141, 142, 143, 144],
                dtype=np.float32,
            ),
        )

    def test_partial_optional_group_is_rejected(self) -> None:
        schema = build_robot_schema(with_head=True)
        payload = status_payload()
        payload["neck_joint_state"] = {"position": [1, 2]}
        self.assertIsNone(schema.parse_state(payload))

    def test_gripper_command_does_not_replace_missing_values_with_zero(self) -> None:
        self.assertEqual(
            parse_gripper_command({"left_gripper_target_joints_position": [0.5]}),
            (0.5, None),
        )


if __name__ == "__main__":
    unittest.main()
