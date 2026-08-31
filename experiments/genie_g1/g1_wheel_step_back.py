#!/usr/bin/env python3
"""Publish backward twist on /mbc/wheel_command (geometry_msgs/TwistStamped).

Open-loop: approximate distance ≈ speed * duration (no odometry closed loop).

Run on G1 as root (Fast DDS SHM with motion-control):

  sudo env -i HOME=/root PATH=/opt/ros/humble/bin:/usr/bin:/bin \\
    bash --norc --noprofile -c 'source /opt/ros/humble/setup.bash && \\
      python3 /tmp/g1_wheel_step_back.py --distance-m 0.5 --speed 0.15'
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import TwistStamped  # type: ignore
        from rclpy.node import Node  # type: ignore
    except ImportError as e:
        print("ROS 2 Python packages missing:", e, file=sys.stderr)
        sys.exit(1)

    env_burst = os.environ.get("G1_WHEEL_BURST_SEC")
    env_dist = os.environ.get("G1_WHEEL_DISTANCE_M")
    env_speed = os.environ.get("G1_WHEEL_SPEED")
    env_bx = os.environ.get("G1_WHEEL_BACKWARD_X")

    p = argparse.ArgumentParser(description="Backward velocity on /mbc/wheel_command (mobile base).")
    p.add_argument(
        "--distance-m",
        type=float,
        default=float(env_dist) if env_dist not in (None, "") else None,
        help="Target backward distance (m), open-loop: duration = distance / speed. Overrides burst duration.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=float(env_speed) if env_speed not in (None, "") else 0.15,
        help="Backward speed magnitude (m/s); publishes linear.x = -speed. Default 0.15.",
    )
    p.add_argument("--hz", type=float, default=float(os.environ.get("G1_WHEEL_HZ", "40")))
    p.add_argument(
        "--burst-sec",
        type=float,
        default=float(env_burst) if env_burst not in (None, "") else 1.0,
        help="Duration to publish twist (s). Ignored if --distance-m is set.",
    )
    p.add_argument(
        "--max-duration-sec",
        type=float,
        default=60.0,
        help="Safety cap on motion duration (default 60).",
    )
    args = p.parse_args()

    speed = abs(float(args.speed))
    if speed < 1e-6:
        print("speed must be > 0", file=sys.stderr)
        sys.exit(2)

    backward_x = -speed
    if env_bx not in (None, ""):
        backward_x = float(env_bx)
        speed = abs(backward_x)

    if args.distance_m is not None:
        burst_sec = abs(float(args.distance_m)) / speed
    else:
        burst_sec = float(args.burst_sec)

    if burst_sec > float(args.max_duration_sec):
        print(
            f"Computed duration {burst_sec:.3f}s exceeds --max-duration-sec={args.max_duration_sec}; aborting.",
            file=sys.stderr,
        )
        sys.exit(3)

    hz = max(1.0, float(args.hz))
    rclpy.init(args=sys.argv)
    node = Node("g1_wheel_step_back_once")
    pub = node.create_publisher(TwistStamped, "/mbc/wheel_command", 10)
    time.sleep(0.3)

    msg = TwistStamped()
    msg.header.frame_id = "base_link"

    period = 1.0 / hz
    n_steps = max(1, int(burst_sec / period))
    print(
        f"Publishing linear.x={backward_x} for ~{burst_sec:.3f}s "
        f"({n_steps} msgs @ {hz} Hz); open-loop distance ~{speed * burst_sec:.3f} m"
    )
    msg.twist.linear.x = float(backward_x)
    for _ in range(n_steps):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        time.sleep(period)

    msg.twist.linear.x = 0.0
    msg.twist.angular.z = 0.0
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)
    print("Published zero twist stop.")
    time.sleep(0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
