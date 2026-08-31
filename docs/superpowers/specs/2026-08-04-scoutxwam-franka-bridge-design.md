# ScoutXWAM DROID-100 → Franka FR3 Bridge Design

Date: 2026-08-04  
Status: approved (user: 按这个做)

## Goal

Connect the H100 ScoutXWAM DROID-100 Franka 5B inference package to the existing lab FR3 stack on `yao@10.229.20.125`, delivering:

1. Persistent HTTP inference service (port **8010**)
2. Sensor dry-run client (no motion publishers)
3. Open-loop motion path with explicit dual-gate arming

## Topology

```text
H100 a25689@10.239.121.11:31126
  scoutxwam_droid100_inference/serve.py → 127.0.0.1:8010
        │ ssh -R 8010:127.0.0.1:8010
        ▼
yao@10.229.20.125  (wired FCI → FR3 10.229.66.91)
  dryrun_franka.py / open_loop_franka.py
```

Constraints:

- Do not modify Franka Desk / robot network settings.
- Do not route FR3 FCI via `10.229.66.70`.
- Port **8010** (leave kairos WAM on **8005** alone).
- Motion default **off**; require `--i-approve-motion` + matching `SCOUT_ARM_TOKEN`.

## Interfaces

### ScoutXWAM proprio / action (native, not LIBERO)

| Field | Shape / layout |
| --- | --- |
| video | `[2, 256, 320, 3]` uint8, order **exterior then wrist** |
| proprio | `[8]` = xyz(3) + quat_xyzw(4) + gripper(1) in ~[0,1] |
| actions | `[32, 7]` denormalized relative eef: Δxyz + Δaxis-angle + gripper |
| proprios | `[9, 16]` predicted padded X-WAM states (logged only) |

### HTTP API (`127.0.0.1:8010`)

- `GET /health` → `{ok, model_loaded, device}`
- `POST /v1/infer` JSON:
  - `prompt: str`
  - `proprio: float[8]`
  - `video_shape: [2,H,W,3]`
  - `video_b64: base64(uint8 raw C-order)`
  - optional `seed`, `action_denoise_steps`
  - response: `{actions, proprios, infer_s}`

### Lab sensors (reuse existing)

- Cams: compressed `/cam2/...` (scene/exterior) + `/cam1/...` (wrist)
- Pose: `/franka_robot_state_broadcaster/current_pose`
- Joints: `/franka/joint_states`
- Gripper: `/franka_gripper/joint_states`

### Motion

- Plan: clamp Δeef → workspace check → FR3 IK (reuse kairos `phase2/ik_fr3.py`, `limits.py`)
- Execute (armed only): `GelloTakeover` → `/gello/joint_states` @ 50 Hz; optional gripper percent topic

## Code layout

`experiments/scoutxwam_franka_bridge/` in FastWAM; `serve.py` also deployed into `/home/a25689/FastWAM/scoutxwam_droid100_inference/` on H100.

## Safety

- Dry-run must never create control publishers.
- Open-loop refuses without `--i-approve-motion` and token match.
- Per-step Δxyz / Δrot clamps and workspace AABB before IK.
- Reject large joint jumps (`dq > 0.35` rad).
