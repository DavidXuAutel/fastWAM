#!/usr/bin/env python3
"""Apply a small delta (degrees) to one arm joint on G1 via /wbc/arm_command.

Reads current /hal/arm_joint_state (14 positions: left7 + right7, radians),
adds delta to one joint, publishes sensor_msgs/JointState to /wbc/arm_command.

Run on the robot with sudo so DDS matches motion-control (same SHM domain as HAL):

  sudo env -i HOME=/root PATH=/opt/ros/humble/bin:/usr/bin:/bin \\
    bash --norc --noprofile -c 'source /opt/ros/humble/setup.bash && \\
      python3 /tmp/g1_arm_joint_delta.py --delta-deg 5 --side left'

Joint indexing
----------------
- ``--side left``: ``--joint-index`` is 0..6 within the left arm (indices 0..6 in the 14-vector).
- ``--side right``: local index maps to global indices 7..13.

Default ``--joint-index 1`` is a guess (often shoulder-related); URDF order varies.
If ``JointState.name`` is populated, prefer ``--joint-name-substr shoulder`` (first match).

Optional: ``--enter-servo`` publishes ``/wbc/set_control_mode`` MODE_SERVO once (needs genie_msgs).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional


def _maybe_enter_servo() -> None:
    try:
        from genie_msgs.msg import SetControlMode  # type: ignore
        import rclpy  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception as exc:
        print("--enter-servo skipped:", exc, file=sys.stderr)
        return

    rclpy.init(args=sys.argv)
    node = Node("g1_arm_delta_set_servo_once")
    pub = node.create_publisher(SetControlMode, "/wbc/set_control_mode", 10)
    msg = SetControlMode()
    msg.header.frame_id = ""
    msg.input_type = 54
    msg.control_mode = 1
    time.sleep(0.2)
    pub.publish(msg)
    node.destroy_node()
    rclpy.shutdown()
    print("Published /wbc/set_control_mode MODE_SERVO (input_type=54).")


def main() -> None:
    try:
        import rclpy  # type: ignore
        from rclpy.node import Node  # type: ignore
        from rclpy.qos import (  # type: ignore
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import JointState  # type: ignore
    except ImportError as e:
        print("ROS 2 Python packages missing:", e, file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description="Delta one arm joint on G1 (/wbc/arm_command).")
    p.add_argument("--delta-deg", type=float, default=5.0, help="Joint delta in degrees.")
    p.add_argument("--side", choices=("left", "right"), default="left")
    p.add_argument(
        "--joint-index",
        type=int,
        default=None,
        help="Joint index within that arm (0..6). Default 1 if name match unused.",
    )
    p.add_argument(
        "--joint-name-substr",
        default=None,
        help="If JointState.name is set, pick first arm joint whose name contains this (case-insensitive).",
    )
    p.add_argument("--topic-state", default="/hal/arm_joint_state")
    p.add_argument("--topic-cmd", default="/wbc/arm_command")
    p.add_argument("--wait-sec", type=float, default=12.0)
    p.add_argument("--repeat", type=int, default=12, help="Republish same command this many times.")
    p.add_argument("--rate-hz", type=float, default=40.0)
    p.add_argument(
        "--enter-servo",
        action="store_true",
        help="Publish MODE_SERVO once before motion (needs genie_msgs + overlay).",
    )
    args = p.parse_args()

    delta_rad = math.radians(float(args.delta_deg))
    arm_dof = 7
    offset = 0 if args.side == "left" else arm_dof

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )

    if args.enter_servo:
        _maybe_enter_servo()

    rclpy.init(args=sys.argv)

    node = Node("g1_arm_joint_delta_once")
    latest_pos: List[float] = []
    latest_names: Optional[List[str]] = None

    def _cb(msg: JointState) -> None:
        nonlocal latest_pos, latest_names
        if msg.position and len(msg.position) >= 2 * arm_dof:
            latest_pos = [float(x) for x in msg.position[: 2 * arm_dof]]
            if msg.name and len(msg.name) >= 2 * arm_dof:
                latest_names = [str(x) for x in msg.name[: 2 * arm_dof]]

    sub = node.create_subscription(JointState, args.topic_state, _cb, qos)
    pub = node.create_publisher(JointState, args.topic_cmd, 10)

    deadline = time.monotonic() + float(args.wait_sec)
    while time.monotonic() < deadline and not latest_pos:
        rclpy.spin_once(node, timeout_sec=0.05)

    if not latest_pos:
        print(f"No data on {args.topic_state} within {args.wait_sec}s.", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(3)

    local_idx: Optional[int] = args.joint_index
    if args.joint_name_substr and latest_names:
        arm_names = latest_names[offset : offset + arm_dof]
        needle = args.joint_name_substr.lower()
        found = None
        for i, nm in enumerate(arm_names):
            if needle in nm.lower():
                found = i
                break
        if found is None:
            print(f"No joint name containing {needle!r} in {arm_names}", file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(4)
        local_idx = found
    if local_idx is None:
        local_idx = 1

    if local_idx < 0 or local_idx >= arm_dof:
        print("--joint-index must be in [0, 6].", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(2)

    gidx = offset + local_idx
    target = latest_pos.copy()
    target[gidx] += delta_rad

    print(
        f"{args.side} arm local joint {local_idx} (global {gidx}): "
        f"{math.degrees(latest_pos[gidx]):.4f}° -> {math.degrees(target[gidx]):.4f}° "
        f"(delta {args.delta_deg}°)"
    )

    period = 1.0 / max(1.0, float(args.rate_hz))
    js = JointState()
    js.header.frame_id = ""
    js.position = target
    if latest_names:
        js.name = latest_names

    for _ in range(max(1, int(args.repeat))):
        js.header.stamp = node.get_clock().now().to_msg()
        pub.publish(js)
        time.sleep(period)

    print(f"Published {args.topic_cmd} x{max(1, int(args.repeat))}.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
