#!/usr/bin/env python3
"""Canonical Autolife state/action schema used by the LeRobot collector.

LeRobot's relative-action processor subtracts state and action by array index,
not by matching feature names.  This module therefore owns both the ordering
and the names for every joint-controlled dimension.  Keeping that information
in one place prevents a parser change from silently corrupting relative actions.

The first 16 dimensions preserve the collector's historical arm/gripper layout.
Optional head and waist groups are appended, so enabling a new observation never
shifts an existing arm or gripper index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointGroup:
    """Describe one contiguous group in the robot's ROS JSON payloads."""

    name: str
    names: tuple[str, ...]
    state_key: str
    status_target_key: str | None
    command_target_key: str | None

    @property
    def size(self) -> int:
        return len(self.names)


# These orders come from Autolife's running arm controller on robot 283.  In
# particular, neck is Roll/Pitch/Yaw and leg_waist is Ankle/Knee/Pitch/Yaw.
LEFT_ARM = JointGroup(
    name="left_arm",
    names=(
        "left_shoulder_inner",
        "left_shoulder_outer",
        "left_upper_arm",
        "left_elbow",
        "left_forearm",
        "left_wrist_upper",
        "left_wrist_lower",
    ),
    state_key="left_arm_joint_state",
    status_target_key="left_arm_target_joint_state",
    command_target_key="left_arm_target_joints_position",
)
RIGHT_ARM = JointGroup(
    name="right_arm",
    names=(
        "right_shoulder_inner",
        "right_shoulder_outer",
        "right_upper_arm",
        "right_elbow",
        "right_forearm",
        "right_wrist_upper",
        "right_wrist_lower",
    ),
    state_key="right_arm_joint_state",
    status_target_key="right_arm_target_joint_state",
    command_target_key="right_arm_target_joints_position",
)
LEFT_GRIPPER = JointGroup(
    name="left_gripper",
    names=("left_gripper",),
    state_key="left_gripper_state",
    status_target_key=None,
    command_target_key=None,
)
RIGHT_GRIPPER = JointGroup(
    name="right_gripper",
    names=("right_gripper",),
    state_key="right_gripper_state",
    status_target_key=None,
    command_target_key=None,
)
HEAD = JointGroup(
    name="head",
    names=("neck_roll", "neck_pitch", "neck_yaw"),
    state_key="neck_joint_state",
    status_target_key="neck_target_joint_state",
    command_target_key="neck_target_joints_position",
)
WAIST = JointGroup(
    name="waist",
    names=("leg_ankle", "leg_knee", "waist_pitch", "waist_yaw"),
    state_key="leg_waist_joint_state",
    status_target_key="leg_waist_target_joint_state",
    command_target_key="leg_waist_target_joints_position",
)

BASE_GROUPS = (LEFT_ARM, RIGHT_ARM, LEFT_GRIPPER, RIGHT_GRIPPER)


def _numeric_list(value: Any, size: int) -> list[float] | None:
    """Return exactly ``size`` numeric values, rejecting partial payloads."""

    if not isinstance(value, (list, tuple)) or len(value) < size:
        return None
    values = value[:size]
    if not all(isinstance(item, (int, float, bool)) for item in values):
        return None
    return [float(item) for item in values]


def _state_positions(payload: Any, group: JointGroup) -> list[float] | None:
    if not isinstance(payload, dict):
        return None
    state = payload.get(group.state_key)
    if not isinstance(state, dict):
        return None
    return _numeric_list(state.get("position"), group.size)


def _top_level_positions(payload: Any, key: str | None, size: int) -> list[float] | None:
    if not isinstance(payload, dict) or key is None:
        return None
    return _numeric_list(payload.get(key), size)


@dataclass(frozen=True)
class RobotSchema:
    """Resolved state/action layout for one recording session.

    For joint and status-target action modes, ``names`` is used for both
    ``observation.state`` and ``action``.  This is the contract required for
    index-wise relative action conversion during training.
    """

    groups: tuple[JointGroup, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for group in self.groups for name in group.names)

    @property
    def size(self) -> int:
        return len(self.names)

    def parse_state(self, payload: Any) -> np.ndarray | None:
        values: list[float] = []
        for group in self.groups:
            group_values = _state_positions(payload, group)
            if group_values is None:
                return None
            values.extend(group_values)
        return np.asarray(values, dtype=np.float32)

    def parse_status_target(self, payload: Any) -> np.ndarray | None:
        """Parse controller targets embedded in the whole-body status topic.

        The status packet does not expose separate gripper targets.  For those
        two dimensions we retain the measured gripper position, matching the
        collector's existing behavior.  Arm/head/waist dimensions use actual
        controller targets and remain index-aligned with ``parse_state``.
        """

        values: list[float] = []
        for group in self.groups:
            if group.status_target_key is None:
                group_values = _state_positions(payload, group)
            else:
                group_values = _top_level_positions(payload, group.status_target_key, group.size)
            if group_values is None:
                return None
            values.extend(group_values)
        return np.asarray(values, dtype=np.float32)

    def parse_body_command(self, payload: Any) -> dict[str, list[float]] | None:
        """Parse all non-gripper groups from the whole-body command topic."""

        result: dict[str, list[float]] = {}
        for group in self.groups:
            if group.command_target_key is None:
                continue
            values = _top_level_positions(payload, group.command_target_key, group.size)
            # Some older publishers reuse the nested status representation.
            if values is None:
                values = _state_positions(payload, group)
            if values is None:
                return None
            result[group.name] = values
        return result

    def compose_command_action(
        self,
        body_groups: dict[str, list[float]],
        grippers: tuple[float | None, float | None],
    ) -> np.ndarray | None:
        """Merge whole-body and gripper topics in canonical schema order."""

        gripper_values = {
            "left_gripper": grippers[0],
            "right_gripper": grippers[1],
        }
        values: list[float] = []
        for group in self.groups:
            if group.name in gripper_values:
                value = gripper_values[group.name]
                if value is None:
                    return None
                values.append(float(value))
                continue
            group_values = body_groups.get(group.name)
            if group_values is None or len(group_values) != group.size:
                return None
            values.extend(group_values)
        return np.asarray(values, dtype=np.float32)


def build_robot_schema(with_head: bool = False, with_waist: bool = False) -> RobotSchema:
    """Build a stable base-16 schema with optional groups appended."""

    groups = list(BASE_GROUPS)
    if with_head:
        groups.append(HEAD)
    if with_waist:
        groups.append(WAIST)
    return RobotSchema(tuple(groups))


def parse_gripper_command(payload: Any) -> tuple[float | None, float | None]:
    """Parse the dedicated gripper command topic without inventing defaults."""

    left = _top_level_positions(payload, "left_gripper_target_joints_position", 1)
    right = _top_level_positions(payload, "right_gripper_target_joints_position", 1)
    return (left[0] if left else None, right[0] if right else None)
