#!/usr/bin/env python3
"""Open-loop: ScoutXWAM actions → clamp/IK → optional /gello/joint_states.

Default is plan-only. Hardware requires:
  export SCOUT_ARM_TOKEN=...
  python open_loop_franka.py --i-approve-motion --arm-token "$SCOUT_ARM_TOKEN"
"""
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


def _add_kairos_phase2() -> Path:
    root = Path(os.environ.get("KAIROS_ROOT", Path.home() / "kairos")).resolve()
    phase2 = root / "scripts" / "phase2"
    if not phase2.is_dir():
        raise SystemExit(f"kairos phase2 not found at {phase2}; set KAIROS_ROOT")
    if str(phase2) not in sys.path:
        sys.path.insert(0, str(phase2))
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout-url", default=os.environ.get("SCOUT_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-denoise-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--step-dt", type=float, default=0.18)
    parser.add_argument("--max-abs-xyz", type=float, default=0.02)
    parser.add_argument("--max-abs-rot", type=float, default=0.05)
    parser.add_argument("--timeout-image-s", type=float, default=30.0)
    parser.add_argument("--plan-only", action="store_true",
                        help="Force plan-only even if arm flags present")
    parser.add_argument("--i-approve-motion", action="store_true")
    parser.add_argument("--arm-token", default="")
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--log-dir", default="")
    args = parser.parse_args()

    # Default: plan-only. Motion only when explicitly armed and not --plan-only.
    plan_only = bool(args.plan_only) or not args.i_approve_motion

    kairos_root = _add_kairos_phase2()
    from arming import ArmingGate
    from gello_takeover import GelloTakeover
    from ik_fr3 import FR3IK, SE3, axisangle_to_rot, pose_to_se3, quat_xyzw_to_rot
    from limits import StepLimits, WorkspaceLimits, clamp_eef_delta, reject_if_outside_workspace

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
    from proprio_scout import build_scout_proprio, scout_gripper_to_percent

    gate = ArmingGate()
    expect = os.environ.get("SCOUT_ARM_TOKEN", "").strip()
    armed = False
    if plan_only:
        print("[open_loop] PLAN ONLY — no hardware commands", flush=True)
    else:
        if not args.i_approve_motion:
            raise SystemExit("refusing: pass --i-approve-motion (or keep plan-only)")
        if not expect:
            raise SystemExit("refusing: set env SCOUT_ARM_TOKEN first")
        if not args.arm_token or args.arm_token != expect:
            raise SystemExit("refusing: --arm-token does not match SCOUT_ARM_TOKEN")
        gate.issue_token(expect)
        gate.arm(args.arm_token)
        gate.require_armed()
        armed = True
        print("[open_loop] ARMED — will command /gello/joint_states", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or Path.home() / "scoutxwam_openloop_logs" / ts)
    log_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = log_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    client = ScoutClient(base_url=args.scout_url)
    health = client.health()
    print(f"[open_loop] health={health} kairos={kairos_root}", flush=True)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Float32

    class Sensors(Node):
        def __init__(self) -> None:
            super().__init__("scoutxwam_open_loop")
            self.cam1 = None
            self.cam2 = None
            self.joints = None
            self.pose = None
            self.grip = None
            for qos in realsense_image_qos():
                self.create_subscription(CompressedImage, CAM1_COMPRESSED, self._on_cam1, qos)
                self.create_subscription(CompressedImage, CAM2_COMPRESSED, self._on_cam2, qos)
            self.create_subscription(JointState, "/franka/joint_states", self._on_j, 10)
            self.create_subscription(
                PoseStamped,
                "/franka_robot_state_broadcaster/current_pose",
                self._on_p,
                qos_profile_sensor_data,
            )
            self.create_subscription(JointState, "/franka_gripper/joint_states", self._on_g, 10)

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

        def _on_j(self, msg):
            self.joints = msg

        def _on_p(self, msg):
            self.pose = msg

        def _on_g(self, msg):
            self.grip = msg

    rclpy.init()
    node = Sensors()
    t_deadline = time.time() + args.timeout_image_s
    while time.time() < t_deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        ready = (
            node.cam1 is not None
            and node.cam2 is not None
            and node.pose is not None
            and node.joints is not None
        )
        if ready:
            break
    else:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit("timeout waiting for cams/pose/joints")

    # Joint vector
    name_to_pos = dict(zip(node.joints.name, node.joints.position))
    q = np.array(
        [float(name_to_pos[f"fr3_joint{i}"]) for i in range(1, 8)],
        dtype=np.float64,
    )
    p = node.pose.pose.position
    o = node.pose.pose.orientation
    pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
    if node.grip is not None and len(node.grip.position) >= 1:
        fingers = np.array(node.grip.position[:2], dtype=np.float32)
    else:
        fingers = np.array([0.04, 0.04], dtype=np.float32)

    ik = FR3IK()
    T_meas = pose_to_se3(pos, quat)
    ik.calibrate_tool_from_measured(q, T_meas)
    R = quat_xyzw_to_rot(quat)
    proprio = build_scout_proprio(pos.astype(np.float32), quat.astype(np.float32), fingers)
    video = stack_exterior_wrist(node.cam2, node.cam1)
    Image.fromarray(video[0]).save(frames_dir / "exterior.png")
    Image.fromarray(video[1]).save(frames_dir / "wrist.png")

    t0 = time.time()
    result = client.infer(
        video=video,
        proprio=proprio,
        prompt=args.prompt,
        seed=args.seed,
        action_denoise_steps=args.action_denoise_steps,
    )
    dt_inf = time.time() - t0
    chunk = np.asarray(result["actions"], dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[-1] < 7:
        raise RuntimeError(f"bad actions shape {chunk.shape}")
    n_steps = min(int(args.max_steps), int(chunk.shape[0]))
    print(f"[open_loop] infer dt={dt_inf:.2f}s chunk={chunk.shape} exec_steps={n_steps}", flush=True)

    limits = StepLimits(max_abs_xyz=args.max_abs_xyz, max_abs_rot=args.max_abs_rot)
    cur_pos = pos.copy()
    cur_R = R.copy()
    cur_q = q.copy()
    plan = []
    for i in range(n_steps):
        # Scout server already denormalizes actions.
        clamped = clamp_eef_delta(chunk[i, :7], limits, mode="linf")
        dlt = clamped.astype(np.float64)
        cur_pos = cur_pos + dlt[0:3]
        cur_R = axisangle_to_rot(dlt[3:6]) @ cur_R
        reject_if_outside_workspace(
            cur_pos,
            WorkspaceLimits(xyz_min=(0.25, -0.45, 0.05), xyz_max=(0.85, 0.45, 0.65)),
        )
        T_des = SE3.from_Rt(cur_R, cur_pos)
        q_des, ok = ik.ik(T_des, cur_q)
        if not ok:
            raise RuntimeError(f"IK failed at step {i}")
        dq = float(np.linalg.norm(q_des - cur_q))
        if dq > 0.35:
            raise RuntimeError(f"joint jump too large at step {i}: dq={dq:.3f} rad")
        plan.append(
            {
                "i": i,
                "raw": chunk[i, :7].tolist(),
                "clamped": clamped.tolist(),
                "ee_xyz": cur_pos.tolist(),
                "q": q_des.tolist(),
                "dq": dq,
                "gripper": float(clamped[-1]),
            }
        )
        cur_q = q_des
        print(
            f"[open_loop] plan[{i}] ee={np.round(cur_pos,4).tolist()} dq={dq:.4f} g={clamped[-1]:.3f}",
            flush=True,
        )

    (log_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    meta = {
        "prompt": args.prompt,
        "armed": armed,
        "plan_only": plan_only,
        "enable_gripper": bool(args.enable_gripper),
        "infer_s": round(dt_inf, 3),
        "n_steps": n_steps,
        "scout_url": args.scout_url,
    }
    (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if plan_only or not armed:
        print(f"[open_loop] DONE plan-only log_dir={log_dir}", flush=True)
        node.destroy_node()
        rclpy.shutdown()
        return 0

    takeover = GelloTakeover(node, rate_hz=50.0)
    takeover.start(q)
    grip_pub = None
    if args.enable_gripper:
        grip_pub = node.create_publisher(
            Float32, "/gripper/gripper_client/target_gripper_width_percent", 10
        )

    try:
        for step in plan:
            takeover.set_goal(np.asarray(step["q"], dtype=np.float64))
            if grip_pub is not None:
                msg = Float32()
                msg.data = scout_gripper_to_percent(step["gripper"])
                grip_pub.publish(msg)
            t_end = time.time() + args.step_dt
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        takeover.stop(resume_gello=False)

    print(f"[open_loop] DONE armed log_dir={log_dir}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
