# ScoutXWAM MuJoCo Sim Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Live-sensor Scout infer → full IK trajectory → MuJoCo-only playback + full disk logs; never command the real arm.

**Architecture:** One script on 125 reuses bridge cameras/client/proprio + kairos IK/limits; drives `mujoco.viewer` with `fr3.mujoco.urdf`; writes `~/scoutxwam_sim_logs/<ts>/`.

**Tech Stack:** Python3, ROS2 Humble, MuJoCo, Scout HTTP `:8010`, kairos phase2 IK.

---

### Task 1: Add `sim_open_loop_mujoco.py`

**Files:**
- Create: `experiments/scoutxwam_franka_bridge/sim_open_loop_mujoco.py`
- Modify: `experiments/scoutxwam_franka_bridge/README.md` (short usage)

**Steps:**
1. Implement script per spec: sensors → infer → plan → MuJoCo play → log all artifacts.
2. Hard-refuse `--i-approve-motion` if present.
3. Sync to `yao@10.229.20.125:~/scoutxwam_franka_bridge/`.
4. Run once with prompt `pick up the pen`, `DISPLAY=:1`, confirm log dir + no `/gello` pubs from script.
5. Verify real EE unchanged vs pre-run snapshot.

**Done when:** Log dir complete; MuJoCo process ran; real robot hold undisturbed.
