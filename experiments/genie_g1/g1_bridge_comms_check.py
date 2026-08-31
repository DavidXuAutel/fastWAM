#!/usr/bin/env python3
"""
Wait for GenieG1TaskEnv observation streams (cameras + /hal/arm_joint_state).

Use after ``source scripts/env_ros2_humble.sh`` and ``source scripts/env_g1_robot.sh``
with the same ROS_DOMAIN_ID / RMW as the robot. Exits 0 only if ``wait_for_observation`` succeeds.

If ``/hal/arm_joint_state`` is published from root over Fast DDS shared memory, a **Mac 客户端**
可能收不到关节流；此时在 **G1 本机** 上运行本脚本（或 ``--arm-only`` 做半栈检查），或按厂商方式对齐 DDS。

Example::

  cd /path/to/FastWAM
  source scripts/env_ros2_humble.sh
  source scripts/env_g1_robot.sh
  python experiments/genie_g1/g1_bridge_comms_check.py --camera-profile hdas --g1-ip 10.229.66.60
  python experiments/genie_g1/g1_bridge_comms_check.py --arm-only --timeout-sec 15 --g1-ip 10.229.66.60
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_g1_remote_link():
    path = PROJECT_ROOT / "experiments" / "genie_g1" / "g1_remote_link.py"
    spec = importlib.util.spec_from_file_location("fastwam_g1_remote_link", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load g1_remote_link from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ros_g1_bridge():
    path = PROJECT_ROOT / "experiments" / "genie_g1" / "ros_g1_bridge.py"
    spec = importlib.util.spec_from_file_location("fastwam_ros_g1_bridge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ros_g1_bridge from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ros2_humble_env():
    path = PROJECT_ROOT / "experiments" / "genie_g1" / "ros2_humble_env.py"
    spec = importlib.util.spec_from_file_location("fastwam_ros2_humble_env", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ros2_humble_env from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROS2 comms check for GenieG1TaskEnv observation topics.")
    p.add_argument("--camera-profile", choices=("gdk", "hdas"), default="gdk")
    p.add_argument("--topic-head-rgb", default="/camera/head_color")
    p.add_argument("--topic-left-rgb", default="/camera/hand_left_color")
    p.add_argument("--topic-right-rgb", default="/camera/hand_right_color")
    p.add_argument("--topic-arm-state", default="/hal/arm_joint_state")
    p.add_argument("--timeout-sec", type=float, default=35.0)
    p.add_argument(
        "--arm-only",
        action="store_true",
        help="Only wait for /hal/arm_joint_state (skip cameras). Useful for DDS checks when images are missing.",
    )
    p.add_argument("--g1-ip", default="10.229.66.60")
    p.add_argument("--no-g1-remote", action="store_true")
    p.add_argument("--remote-dds", choices=("fastrtps", "cyclonedds"), default="fastrtps")
    p.add_argument("--ros-domain-id", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.camera_profile == "hdas":
        args.topic_head_rgb = "/camera/head_center_fisheye"
        args.topic_left_rgb = "/hdas/camera_wrist_left/color/image_rect_raw/compressed"
        args.topic_right_rgb = "/hdas/camera_wrist_right/color/image_rect_raw/compressed"
        cam_head_t = cam_left_t = cam_right_t = "compressed"
    else:
        cam_head_t = cam_left_t = cam_right_t = "raw"

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    link = _load_g1_remote_link()
    if not args.no_g1_remote:
        ip = str(args.g1_ip or "").strip()
        if ip.lower() not in {"", "none", "local"}:
            applied = link.apply_remote_ros_env(ip, domain_id=args.ros_domain_id, dds=args.remote_dds)
            print(link.summarize_connection(ip, applied), flush=True)

    ros_env = _load_ros2_humble_env()
    ros_env.warn_if_ros_not_sourced()
    print(ros_env.ros2_env_summary(), flush=True)

    bridge_mod = _load_ros_g1_bridge()
    GenieG1TaskEnv = bridge_mod.GenieG1TaskEnv

    import rclpy

    rclpy.init(args=sys.argv)
    env = GenieG1TaskEnv(
        instruction=" ",
        topic_head_rgb=args.topic_head_rgb,
        topic_left_rgb=args.topic_left_rgb,
        topic_right_rgb=args.topic_right_rgb,
        topic_arm_joint_state=args.topic_arm_state,
        topic_left_ee_state=None,
        topic_right_ee_state=None,
        camera_head_transport=cam_head_t,
        camera_left_transport=cam_left_t,
        camera_right_transport=cam_right_t,
    )
    try:
        if args.arm_only:
            print(
                f"Waiting up to {args.timeout_sec}s for {args.topic_arm_state} only ...",
                flush=True,
            )
        else:
            print(
                f"Waiting up to {args.timeout_sec}s for "
                f"head={args.topic_head_rgb} left={args.topic_left_rgb} right={args.topic_right_rgb} "
                f"+ {args.topic_arm_state} ...",
                flush=True,
            )
        env.wait_for_observation(
            timeout_sec=float(args.timeout_sec), require_cameras=not args.arm_only
        )
        print("comms_ok: required observation streams received at least once.", flush=True)
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
