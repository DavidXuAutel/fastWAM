# ScoutXWAM → MuJoCo Sim (no real-robot motion)

Date: 2026-08-04  
Status: approved (user: 同意 / 方案 A)

## Goal

On lab PC `yao@10.229.20.125`:

1. Read live Franka cameras + EEF proprio (real robot **hold only**, no motion commands).
2. Call ScoutXWAM HTTP `:8010` for one action chunk.
3. Integrate denormalized relative EEF actions → full absolute EE + joint trajectory (IK).
4. Play the **full** trajectory in a MuJoCo FR3 viewer only.
5. Persist **all** request / infer / plan / playback artifacts to disk.

## Non-goals

- No `GelloTakeover`, no `/gello/joint_states` motion publish from this script.
- No closed-loop re-inference inside MuJoCo (no sim cameras).
- Do not modify Desk / robot network.

## Topology

```text
Real FR3 (hold) ──cams/pose/joints/grip──► sim_open_loop_mujoco.py
                                              │
                                              ├─ POST :8010 /v1/infer
                                              ├─ clamp + IK → plan[T]
                                              ├─ MuJoCo viewer (fr3.mujoco.urdf)
                                              └─ ~/scoutxwam_sim_logs/<ts>/
```

## Interfaces (unchanged Scout contract)

| Field | Layout |
| --- | --- |
| video | `[2,256,320,3]` uint8 exterior then wrist |
| proprio | `[8]` xyz + quat_xyzw + gripper |
| actions | `[32,7]` Δxyz + Δaxis-angle + gripper (denorm) |

## Script

`experiments/scoutxwam_franka_bridge/sim_open_loop_mujoco.py` (sync to `~/scoutxwam_franka_bridge/` on 125).

Behavior:

- Subscribe ROS cams/pose/joints/gripper (same topics as `open_loop_franka.py`).
- Infer via `ScoutClient`.
- Build plan with kairos `clamp_eef_delta` / workspace / `FR3IK` (same as open-loop).
- Launch MuJoCo passive viewer; set `qpos` per step at `--step-dt`; include fingers from gripper scalar.
- Never create control publishers.

## Logging layout

`~/scoutxwam_sim_logs/<YYYYMMDD_HHMMSS>/`:

- `frames/exterior.png`, `frames/wrist.png`
- `request.json` — prompt, proprio, seed, scout_url, action_denoise_steps
- `actions.npy`, `actions.json` — raw denormalized chunk
- `proprios.json` — model predicted proprios if present
- `plan.json` — per-step raw, clamped, ee_xyz, q, gripper, dq
- `playback.jsonl` — wall time, step index, q applied in MuJoCo
- `meta.json` — health, infer_s, n_steps, model path, host, DISPLAY, MUJOCO_MODEL

## Safety

- Default and only mode: sim playback.
- Refuse if `--i-approve-motion` is passed (wrong script; use `open_loop_franka.py`).
- Leave existing `scout_hold_home` alone (do not kill / takeover).

## Success criteria

- Viewer shows full chunk motion matching `plan.json` joints.
- Real robot EE unchanged during run (hold maintained by external hold process).
- Log dir contains all files above after one run.
