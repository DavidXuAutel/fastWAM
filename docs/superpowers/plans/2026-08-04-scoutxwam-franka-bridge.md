# ScoutXWAM Franka Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship HTTP serve + dry-run + gated open-loop bridge from ScoutXWAM H100 to FR3 on 125.

**Architecture:** FastAPI on H100 `:8010`; reverse tunnel to lab PC; ROS2 clients reuse kairos phase2 IK/takeover; ScoutXWAM native quat proprio.

**Tech Stack:** FastAPI/uvicorn, PyTorch (mot-wam), ROS2 Humble on 125, numpy/PIL.

---

### Task 1: Package scaffolding + HTTP client/server

**Files:**
- Create: `experiments/scoutxwam_franka_bridge/{__init__.py,serve.py,client.py,proprio_scout.py,cameras.py,README.md,start_serve.sh,tunnel_to_125.sh}`

**Steps:**
1. Implement `serve.py` loading ScoutXWAM once, exposing `/health` and `/v1/infer`.
2. Implement `client.py` JSON+base64 client with timeout.
3. Implement camera resize to `[2,256,320,3]` and quat proprio builder.
4. Deploy `serve.py` to H100 package root; start after smoke frees GPU0.
5. Verify `curl /health` on H100.

### Task 2: Dry-run client (no motion)

**Files:**
- Create: `experiments/scoutxwam_franka_bridge/dryrun_franka.py`

**Steps:**
1. Subscribe cams/pose/gripper on 125; build Scout request; call `:8010`.
2. Log frames + `actions.jsonl`; assert no control publishers created.
3. Document run command in README.

### Task 3: Open-loop gated motion

**Files:**
- Create: `experiments/scoutxwam_franka_bridge/open_loop_franka.py`

**Steps:**
1. Reuse kairos `limits`, `ik_fr3`, `gello_takeover`, `arming` via `KAIROS_ROOT`.
2. Default plan-only; arm only with `--i-approve-motion` + `SCOUT_ARM_TOKEN`.
3. Clamp already-denormalized Scout actions; IK; optional execute.

### Task 4: Tunnel + deploy notes

**Files:**
- `tunnel_to_125.sh`, README deploy section

**Steps:**
1. Mirror kairos tunnel pattern for port 8010.
2. Sync bridge scripts to 125 `~/scoutxwam_franka_bridge` (or FastWAM path).
