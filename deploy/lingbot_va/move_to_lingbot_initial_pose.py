#!/usr/bin/env python3
"""Move AutoLife arms slowly to the LingBot training-set start-pose median."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR.parent
REPO_DIR = DEPLOY_DIR.parent
COLLECTOR_DIR = REPO_DIR / "lerobot_data_collector"
for import_dir in (DEPLOY_DIR, REPO_DIR, COLLECTOR_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from pi05.deploy_pi05 import AutoLifeCommandNode, JointTelemetryNode, topic_id
from robot_schema import build_robot_schema, schema_state_from_whole_body, whole_body_from_schema_action


# Median of observation.state at frame_index == 0 across the 50 training episodes.
# Layout: left arm 7, right arm 7, left gripper, right gripper.
LINGBOT_START_MEDIAN = np.asarray(
    [
        25.593,
        -7.075,
        5.548,
        113.930,
        -3.582,
        -7.114,
        -2.918,
        -21.093,
        13.889,
        -0.835,
        -110.719,
        7.158,
        -2.874,
        -1.282,
        11.792,
        12.721,
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Permit physical motion; default is dry-run.")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-arm-step", type=float, default=0.10, help="Maximum arm-joint degrees per command.")
    parser.add_argument("--max-total-arm-delta", type=float, default=20.0)
    parser.add_argument("--state-timeout-sec", type=float, default=5.0)
    parser.add_argument("--start-delay-sec", type=float, default=3.0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.hz, args.max_arm_step, args.max_total_arm_delta, args.state_timeout_sec) <= 0:
        raise SystemExit("Rate, step, delta, and timeout arguments must be positive.")

    rclpy.init()
    telemetry = JointTelemetryNode(
        f"/topic_arm_whole_body_and_gripper_current_joints_status_{topic_id()}"
    )
    command = AutoLifeCommandNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(telemetry)
    executor.add_node(command)
    schema = build_robot_schema(with_head=False, with_waist=False)

    try:
        deadline = time.monotonic() + args.state_timeout_sec
        current_q23 = None
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            current_q23, received_sec = telemetry.snapshot()
            if current_q23 is not None and time.time() - received_sec <= 0.2:
                break
        if current_q23 is None:
            raise RuntimeError("No fresh complete robot joint state was received.")

        current_schema = schema_state_from_whole_body(current_q23, schema)
        arm_delta = LINGBOT_START_MEDIAN[:14] - current_schema[:14]
        max_index = int(np.argmax(np.abs(arm_delta)))
        max_delta = float(abs(arm_delta[max_index]))
        print("current schema:", np.round(current_schema, 3).tolist())
        print("target schema: ", np.round(LINGBOT_START_MEDIAN, 3).tolist())
        print(f"maximum arm delta: {max_delta:.3f} deg at schema index {max_index}")
        if max_delta > args.max_total_arm_delta:
            raise RuntimeError(
                f"Required arm delta {max_delta:.3f} exceeds --max-total-arm-delta "
                f"{args.max_total_arm_delta:.3f}."
            )
        if not args.publish:
            print("DRY-RUN: no command was published.")
            return

        steps = max(1, math.ceil(max_delta / args.max_arm_step))
        period = 1.0 / args.hz
        print(
            f"Publishing {steps} interpolated commands over about {steps * period:.1f}s; "
            f"starting in {args.start_delay_sec:.1f}s. Press Ctrl-C or emergency stop to abort."
        )
        time.sleep(args.start_delay_sec)
        started = time.monotonic()
        for step in range(1, steps + 1):
            alpha = step / steps
            target_schema = current_schema + alpha * (LINGBOT_START_MEDIAN - current_schema)
            target_q23 = whole_body_from_schema_action(target_schema, current_q23, schema)
            command.publish_action(target_q23, gripper_int=True)
            executor.spin_once(timeout_sec=0.0)
            deadline = started + step * period
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        settle_deadline = time.monotonic() + args.settle_sec
        while time.monotonic() < settle_deadline:
            executor.spin_once(timeout_sec=0.05)
        final_q23, received_sec = telemetry.snapshot()
        if final_q23 is None or time.time() - received_sec > 0.2:
            raise RuntimeError("Motion completed, but final joint telemetry is stale.")
        final_schema = schema_state_from_whole_body(final_q23, schema)
        final_error = np.abs(final_schema[:14] - LINGBOT_START_MEDIAN[:14])
        print("final schema:  ", np.round(final_schema, 3).tolist())
        print(f"maximum final arm error: {float(final_error.max()):.3f} deg")
        if float(final_error.max()) > args.tolerance:
            raise RuntimeError(
                f"Robot did not converge within --tolerance {args.tolerance:.3f} deg."
            )
        print("LingBot initial pose reached.")
    finally:
        executor.shutdown()
        telemetry.destroy_node()
        command.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
