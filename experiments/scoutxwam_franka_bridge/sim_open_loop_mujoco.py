#!/usr/bin/env python3
"""Live Scout infer → full IK trajectory → MuJoCo playback only (no real-robot cmds).

Persists frames, request, actions, plan, playback, and meta under
~/scoutxwam_sim_logs/<ts>/ by default.

This script MUST NOT publish arm/gripper commands. For real motion use
open_loop_franka.py with dual arming.
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
    parser.add_argument("--prompt", default="pick up the pen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-denoise-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=32, help="Play up to this many actions (default full chunk).")
    parser.add_argument("--step-dt", type=float, default=0.12)
    parser.add_argument("--max-abs-xyz", type=float, default=0.02)
    parser.add_argument("--max-abs-rot", type=float, default=0.05)
    parser.add_argument("--timeout-image-s", type=float, default=30.0)
    parser.add_argument(
        "--mujoco-model",
        default=os.environ.get("MUJOCO_MODEL", "/home/yao/franka_mujoco_sync/fr3.mujoco.urdf"),
    )
    parser.add_argument("--log-dir", default="")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Apply qpos offline (still logs playback); skip interactive viewer.",
    )
    parser.add_argument(
        "--i-approve-motion",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.i_approve_motion:
        raise SystemExit(
            "refusing: this is the MuJoCo-only script. "
            "Use open_loop_franka.py for real-robot motion."
        )

    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "glfw"))
    os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":1"))

    kairos_root = _add_kairos_phase2()
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

    import mujoco
    import mujoco.viewer

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or Path.home() / "scoutxwam_sim_logs" / ts)
    log_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = log_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    print(f"[sim] log_dir={log_dir} DISPLAY={os.environ.get('DISPLAY')} model={args.mujoco_model}", flush=True)

    client = ScoutClient(base_url=args.scout_url)
    health = client.health()
    print(f"[sim] health={health} kairos={kairos_root}", flush=True)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, JointState

    class Sensors(Node):
        def __init__(self) -> None:
            super().__init__("scoutxwam_sim_mujoco")
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

    ee0 = {"xyz": pos.tolist(), "quat_xyzw": quat.tolist(), "q": q.tolist(), "fingers": fingers.tolist()}
    print(f"[sim] EE0={np.round(pos,4).tolist()} (real robot hold — no cmds from this script)", flush=True)

    ik = FR3IK()
    T_meas = pose_to_se3(pos, quat)
    ik.calibrate_tool_from_measured(q, T_meas)
    R = quat_xyzw_to_rot(quat)
    proprio = build_scout_proprio(pos.astype(np.float32), quat.astype(np.float32), fingers)
    video = stack_exterior_wrist(node.cam2, node.cam1)
    Image.fromarray(video[0]).save(frames_dir / "exterior.png")
    Image.fromarray(video[1]).save(frames_dir / "wrist.png")

    request = {
        "prompt": args.prompt,
        "proprio": proprio.tolist(),
        "seed": args.seed,
        "scout_url": args.scout_url,
        "action_denoise_steps": args.action_denoise_steps,
        "video_shape": list(video.shape),
        "ee0": ee0,
    }
    (log_dir / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")

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
    np.save(log_dir / "actions.npy", chunk)
    (log_dir / "actions.json").write_text(
        json.dumps(chunk.tolist(), indent=2), encoding="utf-8"
    )
    if result.get("proprios") is not None:
        (log_dir / "proprios.json").write_text(
            json.dumps(result["proprios"], indent=2), encoding="utf-8"
        )

    n_steps = min(int(args.max_steps), int(chunk.shape[0]))
    print(f"[sim] infer dt={dt_inf:.2f}s chunk={chunk.shape} play_steps={n_steps}", flush=True)

    limits = StepLimits(max_abs_xyz=args.max_abs_xyz, max_abs_rot=args.max_abs_rot)
    workspace = WorkspaceLimits(xyz_min=(0.25, -0.45, 0.02), xyz_max=(0.85, 0.45, 0.65))
    cur_pos = pos.copy()
    cur_R = R.copy()
    cur_q = q.copy()
    plan = []
    plan_stop_reason = None
    for i in range(n_steps):
        clamped = clamp_eef_delta(chunk[i, :7], limits, mode="linf")
        dlt = clamped.astype(np.float64)
        next_pos = cur_pos + dlt[0:3]
        next_R = axisangle_to_rot(dlt[3:6]) @ cur_R
        try:
            reject_if_outside_workspace(next_pos, workspace)
        except ValueError as exc:
            plan_stop_reason = f"workspace at step {i}: {exc}"
            print(f"[sim] WARN truncate plan: {plan_stop_reason}", flush=True)
            break
        T_des = SE3.from_Rt(next_R, next_pos)
        q_des, ok = ik.ik(T_des, cur_q)
        if not ok:
            plan_stop_reason = f"IK failed at step {i}"
            print(f"[sim] WARN truncate plan: {plan_stop_reason}", flush=True)
            break
        dq = float(np.linalg.norm(q_des - cur_q))
        if dq > 0.35:
            plan_stop_reason = f"joint jump at step {i}: dq={dq:.3f}"
            print(f"[sim] WARN truncate plan: {plan_stop_reason}", flush=True)
            break
        cur_pos = next_pos
        cur_R = next_R
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
            f"[sim] plan[{i}] ee={np.round(cur_pos,4).tolist()} dq={dq:.4f} g={clamped[-1]:.3f}",
            flush=True,
        )
    if not plan:
        raise RuntimeError(f"empty plan ({plan_stop_reason or 'unknown'})")

    (log_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    model_path = Path(args.mujoco_model)
    if not model_path.is_file():
        raise SystemExit(f"MuJoCo model not found: {model_path}")

    joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    finger_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
    finger_open = 0.04

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_to_qpos = {}
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint missing in model: {name}")
        joint_to_qpos[name] = int(model.jnt_qposadr[jid])
    finger_to_qpos = {}
    for name in finger_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            finger_to_qpos[name] = int(model.jnt_qposadr[jid])

    def apply_q(q7: np.ndarray, grip: float) -> None:
        for i, name in enumerate(joint_names):
            data.qpos[joint_to_qpos[name]] = float(q7[i])
        finger = float(np.clip(scout_gripper_to_percent(grip), 0.0, 1.0)) * finger_open
        for adr in finger_to_qpos.values():
            data.qpos[adr] = finger
        mujoco.mj_forward(model, data)

    # Start at measured pose
    apply_q(q, float(proprio[7]))
    playback_path = log_dir / "playback.jsonl"
    playback_f = playback_path.open("w", encoding="utf-8")

    def log_playback(step: int, q7: np.ndarray, grip: float, phase: str) -> None:
        rec = {
            "t_wall": time.time(),
            "step": step,
            "phase": phase,
            "q": [float(x) for x in q7],
            "gripper": float(grip),
        }
        playback_f.write(json.dumps(rec) + "\n")
        playback_f.flush()

    log_playback(-1, q, float(proprio[7]), "start_measured")

    def play_steps(viewer=None) -> None:
        for step in plan:
            q7 = np.asarray(step["q"], dtype=np.float64)
            grip = float(step["gripper"])
            apply_q(q7, grip)
            log_playback(int(step["i"]), q7, grip, "play")
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    print("[sim] viewer closed early", flush=True)
                    break
            t_end = time.time() + args.step_dt
            while time.time() < t_end:
                if viewer is not None:
                    rclpy.spin_once(node, timeout_sec=0.0)
                else:
                    time.sleep(min(0.01, max(0.0, t_end - time.time())))
            print(f"[sim] play[{step['i']}] q0={q7[0]:.3f} ee={np.round(step['ee_xyz'],4).tolist()}", flush=True)

    try:
        if args.no_viewer:
            print("[sim] --no-viewer: stepping qpos without GUI", flush=True)
            play_steps(None)
        else:
            print("[sim] launching MuJoCo viewer (sim only)", flush=True)
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # brief hold at start
                t_hold = time.time() + 0.5
                while time.time() < t_hold and viewer.is_running():
                    viewer.sync()
                    rclpy.spin_once(node, timeout_sec=0.01)
                play_steps(viewer)
                # hold final briefly
                t_hold = time.time() + 1.0
                while time.time() < t_hold and viewer.is_running():
                    viewer.sync()
                    rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        playback_f.close()

    # Confirm real EE unchanged (best-effort)
    ee1 = None
    t_end = time.time() + 2.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.pose is not None:
        p1 = node.pose.pose.position
        ee1 = [p1.x, p1.y, p1.z]
        drift = float(np.linalg.norm(np.asarray(ee1) - pos))
        print(f"[sim] real EE1={np.round(ee1,4).tolist()} drift={drift:.4f} m", flush=True)

    meta = {
        "prompt": args.prompt,
        "mode": "mujoco_sim_only",
        "infer_s": round(dt_inf, 3),
        "n_steps_requested": n_steps,
        "n_steps_played": len(plan),
        "plan_stop_reason": plan_stop_reason,
        "step_dt": args.step_dt,
        "scout_url": args.scout_url,
        "health": health,
        "mujoco_model": str(model_path),
        "display": os.environ.get("DISPLAY"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "no_viewer": bool(args.no_viewer),
        "ee0": ee0,
        "ee1_real": ee1,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "log_dir": str(log_dir),
    }
    (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[sim] DONE log_dir={log_dir}", flush=True)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
