#!/usr/bin/env python3
"""Dry-run: FR3 sensors → ScoutXWAM HTTP → log only. Never publishes motion."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Safety hard-lock
ENABLE_MOTION = False

# Lab serials after 2026-07-23 swap: cam1/wrist=D435, cam2/exterior=D435I
DEFAULT_WRIST_SERIAL = "141722071359"
DEFAULT_EXTERIOR_SERIAL = "247122072824"


def _grab_realsense_pair(
    *,
    exterior_serial: str,
    wrist_serial: str,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    import pyrealsense2 as rs

    def one(serial: str) -> np.ndarray:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        pipe.start(cfg)
        try:
            frames = pipe.wait_for_frames(5000)
            color = frames.get_color_frame()
            if color is None:
                raise RuntimeError(f"no color frame from {serial}")
            return np.asanyarray(color.get_data()).copy()
        finally:
            pipe.stop()

    return one(exterior_serial), one(wrist_serial)


def main() -> int:
    if ENABLE_MOTION:
        raise RuntimeError("ENABLE_MOTION must stay False in dry-run")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout-url", default=os.environ.get("SCOUT_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--num-infer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-denoise-steps", type=int, default=10)
    parser.add_argument("--timeout-image-s", type=float, default=30.0)
    parser.add_argument("--log-dir", default="")
    parser.add_argument(
        "--camera-backend",
        choices=("ros", "realsense"),
        default=os.environ.get("SCOUT_CAMERA_BACKEND", "ros"),
        help="ros=compressed topics; realsense=pyrealsense2 SDK (bypasses DDS)",
    )
    parser.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--exterior-serial", default=DEFAULT_EXTERIOR_SERIAL)
    parser.add_argument(
        "--proprio",
        default="",
        help="Comma-separated 8 floats (xyz+quat_xyzw+gripper).",
    )
    parser.add_argument(
        "--allow-synthetic-proprio",
        action="store_true",
        help="Use fixed example proprio when live pose unavailable.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from cameras import (
        CAM1_COMPRESSED,
        CAM2_COMPRESSED,
        decode_compressed_image,
        realsense_image_qos,
        stack_exterior_wrist,
    )
    from client import ScoutClient
    from proprio_scout import build_scout_proprio

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or Path.home() / "scoutxwam_dryrun_logs" / ts)
    log_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = log_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    jsonl_path = log_dir / "actions.jsonl"

    client = ScoutClient(base_url=args.scout_url)
    health = client.health()
    print(f"[dryrun] health={health}", flush=True)
    if not health.get("model_loaded"):
        raise SystemExit("Scout service reports model_loaded=false")

    proprio_source = "ros"
    fixed_proprio = None
    if args.proprio:
        fixed_proprio = np.asarray([float(x) for x in args.proprio.split(",")], dtype=np.float32)
        if fixed_proprio.size != 8:
            raise SystemExit("--proprio must have 8 values")
        proprio_source = "cli"
    elif args.allow_synthetic_proprio:
        fixed_proprio = np.asarray(
            [0.4524246, 0.08397451, 0.4086674, 0.10382187, -0.9937578, -0.00813043, 0.04000497, 0.0],
            dtype=np.float32,
        )
        proprio_source = "synthetic_example"

    cam1 = cam2 = None
    pose = None
    grip = None
    node = None

    if args.camera_backend == "realsense":
        print(
            f"[dryrun] camera-backend=realsense exterior={args.exterior_serial} wrist={args.wrist_serial}",
            flush=True,
        )
        cam2, cam1 = _grab_realsense_pair(
            exterior_serial=args.exterior_serial,
            wrist_serial=args.wrist_serial,
        )
        if fixed_proprio is None:
            raise SystemExit(
                "realsense backend needs --proprio or --allow-synthetic-proprio "
                "(live pose requires FCI/ROS)"
            )
    else:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, JointState

        class SensorBuffer(Node):
            def __init__(self) -> None:
                super().__init__("scoutxwam_dryrun")
                self.cam1 = None
                self.cam2 = None
                self.pose = None
                self.grip = None
                for qos in realsense_image_qos():
                    self.create_subscription(CompressedImage, CAM1_COMPRESSED, self._on_cam1, qos)
                    self.create_subscription(CompressedImage, CAM2_COMPRESSED, self._on_cam2, qos)
                self.create_subscription(
                    PoseStamped,
                    "/franka_robot_state_broadcaster/current_pose",
                    self._on_pose,
                    qos_profile_sensor_data,
                )
                self.create_subscription(JointState, "/franka_gripper/joint_states", self._on_grip, 10)

            def _on_cam1(self, msg):
                try:
                    self.cam1 = decode_compressed_image(msg)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"cam1: {exc}")

            def _on_cam2(self, msg):
                try:
                    self.cam2 = decode_compressed_image(msg)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"cam2: {exc}")

            def _on_pose(self, msg):
                self.pose = msg

            def _on_grip(self, msg):
                self.grip = msg

        rclpy.init()
        node = SensorBuffer()
        t_deadline = time.time() + args.timeout_image_s
        need_pose = fixed_proprio is None
        while time.time() < t_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            cams_ok = node.cam1 is not None and node.cam2 is not None
            pose_ok = (not need_pose) or (node.pose is not None)
            if cams_ok and pose_ok:
                break
        else:
            node.destroy_node()
            rclpy.shutdown()
            raise SystemExit(
                "timeout waiting for cams/pose via ROS; "
                "try --camera-backend realsense --allow-synthetic-proprio"
            )
        cam1, cam2 = node.cam1, node.cam2
        pose, grip = node.pose, node.grip

    meta = {
        "prompt": args.prompt,
        "scout_url": args.scout_url,
        "enable_motion": False,
        "num_infer": args.num_infer,
        "camera_backend": args.camera_backend,
        "proprio_source": proprio_source,
    }
    (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for i in range(args.num_infer):
            if args.camera_backend == "realsense" and i > 0:
                cam2, cam1 = _grab_realsense_pair(
                    exterior_serial=args.exterior_serial,
                    wrist_serial=args.wrist_serial,
                )
            elif node is not None:
                import rclpy

                rclpy.spin_once(node, timeout_sec=0.05)
                cam1, cam2 = node.cam1, node.cam2
                pose, grip = node.pose, node.grip

            if fixed_proprio is not None:
                proprio = fixed_proprio.copy()
            else:
                assert pose is not None
                p = pose.pose.position
                q = pose.pose.orientation
                xyz = np.array([p.x, p.y, p.z], dtype=np.float32)
                quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
                if grip is not None and len(grip.position) >= 1:
                    fingers = np.array(grip.position[:2], dtype=np.float32)
                else:
                    fingers = np.array([0.04, 0.04], dtype=np.float32)
                proprio = build_scout_proprio(xyz, quat, fingers)

            video = stack_exterior_wrist(cam2, cam1)
            Image.fromarray(video[0]).save(frames_dir / f"exterior_{i:03d}.png")
            Image.fromarray(video[1]).save(frames_dir / f"wrist_{i:03d}.png")

            t0 = time.time()
            result = client.infer(
                video=video,
                proprio=proprio,
                prompt=args.prompt,
                seed=args.seed + i,
                action_denoise_steps=args.action_denoise_steps,
            )
            dt = time.time() - t0
            record = {
                "i": i,
                "proprio": proprio.tolist(),
                "proprio_source": proprio_source,
                "infer_s": round(dt, 3),
                "server_infer_s": result.get("infer_s"),
                "actions_shape": result.get("actions_shape"),
                "actions0": (result.get("actions") or [[]])[0],
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            print(
                f"[dryrun] i={i} dt={dt:.2f}s actions={result.get('actions_shape')} "
                f"a0={np.round(record['actions0'], 4).tolist() if record['actions0'] else None}",
                flush=True,
            )

    if node is not None:
        import rclpy

        node.destroy_node()
        rclpy.shutdown()
    print(f"[dryrun] DONE log_dir={log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
