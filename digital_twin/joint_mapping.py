"""Pure-Python parsing and joint mapping for the AutoLife digital twin."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


PHYSICAL_JOINT_NAMES: tuple[str, ...] = (
    "Joint_Ankle",
    "Joint_Knee",
    "Joint_Waist_Pitch",
    "Joint_Waist_Yaw",
    "Joint_Left_Shoulder_Inner",
    "Joint_Left_Shoulder_Outer",
    "Joint_Left_UpperArm",
    "Joint_Left_Elbow",
    "Joint_Left_Forearm",
    "Joint_Left_Wrist_Upper",
    "Joint_Left_Wrist_Lower",
    "Joint_Right_Shoulder_Inner",
    "Joint_Right_Shoulder_Outer",
    "Joint_Right_UpperArm",
    "Joint_Right_Elbow",
    "Joint_Right_Forearm",
    "Joint_Right_Wrist_Upper",
    "Joint_Right_Wrist_Lower",
    "Joint_Left_Gripper",
    "Joint_Right_Gripper",
    "Joint_Neck_Roll",
    "Joint_Neck_Pitch",
    "Joint_Neck_Yaw",
)

STATE_GROUPS: tuple[tuple[str, int], ...] = (
    ("leg_waist_joint_state", 4),
    ("left_arm_joint_state", 7),
    ("right_arm_joint_state", 7),
    ("left_gripper_state", 1),
    ("right_gripper_state", 1),
    ("neck_joint_state", 3),
)

TARGET_GROUPS: tuple[tuple[str | None, str, int], ...] = (
    ("leg_waist_target_joint_state", "leg_waist_joint_state", 4),
    ("left_arm_target_joint_state", "left_arm_joint_state", 7),
    ("right_arm_target_joint_state", "right_arm_joint_state", 7),
    (None, "left_gripper_state", 1),
    (None, "right_gripper_state", 1),
    ("neck_target_joint_state", "neck_joint_state", 3),
)


@dataclass(frozen=True)
class JointSample:
    """One complete controller sample in physical q23 order."""

    positions_deg: tuple[float, ...]

    def positions_rad(self) -> tuple[float, ...]:
        return tuple(math.radians(value) for value in self.positions_deg)

    def urdf_positions_rad(self) -> dict[str, float]:
        return dict(zip(PHYSICAL_JOINT_NAMES, self.positions_rad(), strict=True))


def _numeric_values(value: Any, count: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < count:
        return None
    values = value[:count]
    if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in values):
        return None
    return [float(item) for item in values]


def _state_values(payload: dict[str, Any], key: str, count: int) -> list[float] | None:
    group = payload.get(key)
    if not isinstance(group, dict):
        return None
    return _numeric_values(group.get("position"), count)


def parse_status_payload(payload: Any, source: str = "measured") -> JointSample | None:
    """Parse measured or controller-target positions from a ROS status packet.

    Gripper targets are not present in the combined status packet, so target
    mode intentionally mirrors their measured positions.
    """

    if not isinstance(payload, dict):
        return None
    values: list[float] = []
    if source == "measured":
        for state_key, count in STATE_GROUPS:
            group_values = _state_values(payload, state_key, count)
            if group_values is None:
                return None
            values.extend(group_values)
    elif source == "target":
        for target_key, state_key, count in TARGET_GROUPS:
            group_values = (
                _numeric_values(payload.get(target_key), count)
                if target_key is not None
                else None
            )
            if group_values is None:
                group_values = _state_values(payload, state_key, count)
            if group_values is None:
                return None
            values.extend(group_values)
    else:
        raise ValueError(f"Unknown source {source!r}; expected 'measured' or 'target'.")
    return JointSample(tuple(values))


def parse_status_json(text: str, source: str = "measured") -> JointSample | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parse_status_payload(payload, source=source)


def with_mimic_positions(
    primary_positions: dict[str, float],
    mimic_rules: dict[str, tuple[str, float, float]],
) -> dict[str, float]:
    """Expand URDF mimic joints, resolving chains and rejecting cycles."""

    result = dict(primary_positions)
    pending = dict(mimic_rules)
    while pending:
        progressed = False
        for name, (parent, multiplier, offset) in list(pending.items()):
            if parent not in result:
                continue
            result[name] = result[parent] * multiplier + offset
            del pending[name]
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise ValueError(f"Unresolved or cyclic URDF mimic joints: {unresolved}")
    return result
