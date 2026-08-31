#!/usr/bin/env python3
"""
Temporary relay: cosine internal color streams -> ROS2 compressed topics.

This script is designed for G1 environments where cosine publishes
`/camera/hand_left_color` and `/camera/hand_right_color` internally, but
forwarder does not expose them to ROS2.

Prerequisites on robot:
- `cosine_bus_py.py` and its shared lib `libcosine_bus_py.so`
- ROS2 Humble environment sourced
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "ROS2 Python dependencies are missing. Source ROS2 and install sensor_msgs."
    ) from exc


def _load_cosine_subscriber(cosine_py_dir: str):
    if cosine_py_dir not in sys.path:
        sys.path.insert(0, cosine_py_dir)
    try:
        from cosine_bus_py import CosineImageSubscriber  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Cannot import cosine_bus_py from {cosine_py_dir}. "
            "Check path and deployment image."
        ) from exc
    return CosineImageSubscriber


class RelayNode(Node):
    def __init__(
        self,
        *,
        left_src: str,
        right_src: str,
        left_dst: str,
        right_dst: str,
        cosine_py_dir: str,
        cosine_py_lib: Optional[str],
        hz: float,
    ) -> None:
        super().__init__("cosine_color_relay")
        if cosine_py_lib:
            os.environ["COSINE_BUS_PY_LIB"] = cosine_py_lib

        CosineImageSubscriber = _load_cosine_subscriber(cosine_py_dir)
        try:
            self._left_sub = CosineImageSubscriber(left_src)
            self._right_sub = CosineImageSubscriber(right_src)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create cosine subscribers. "
                "Most common cause: missing libcosine_bus_py.so."
            ) from exc

        self._left_pub = self.create_publisher(CompressedImage, left_dst, 10)
        self._right_pub = self.create_publisher(CompressedImage, right_dst, 10)
        self._period = 1.0 / max(hz, 1.0)
        self._left_src = left_src
        self._right_src = right_src

    def _read_and_publish(self, sub, pub) -> bool:
        out = sub.read_latest_frame()
        if not out:
            return False
        data, size, handle = out
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = data[:size]
        pub.publish(msg)
        sub.end_frame_read(handle)
        return True

    def spin_forever(self) -> None:
        self.get_logger().info(
            "Starting relay | left=%s right=%s",
            self._left_src,
            self._right_src,
        )
        while rclpy.ok():
            left_ok = self._read_and_publish(self._left_sub, self._left_pub)
            right_ok = self._read_and_publish(self._right_sub, self._right_pub)
            if not left_ok and not right_ok:
                time.sleep(self._period)
            rclpy.spin_once(self, timeout_sec=0.0)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relay cosine hand color to ROS2 topics")
    p.add_argument("--left-src", default="/camera/hand_left_color")
    p.add_argument("--right-src", default="/camera/hand_right_color")
    p.add_argument("--left-dst", default="/camera/hand_left_color")
    p.add_argument("--right-dst", default="/camera/hand_right_color")
    p.add_argument("--cosine-py-dir", default="/home/agi/app/python/cosine_bus_py")
    p.add_argument("--cosine-py-lib", default=None)
    p.add_argument("--hz", type=float, default=30.0)
    return p.parse_args()


def main() -> None:
    args = _args()
    rclpy.init(args=sys.argv)
    try:
        node = RelayNode(
            left_src=args.left_src,
            right_src=args.right_src,
            left_dst=args.left_dst,
            right_dst=args.right_dst,
            cosine_py_dir=args.cosine_py_dir,
            cosine_py_lib=args.cosine_py_lib,
            hz=args.hz,
        )
    except Exception as exc:
        rclpy.shutdown()
        raise
    try:
        node.spin_forever()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
