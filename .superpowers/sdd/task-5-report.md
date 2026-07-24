# Task 5 Report: Oracle / Pilot Gates and Collection Launch

## Status

Implemented and locally verified. The real OpenFly oracle/pilot and DAgger run
remain blocked until the approved `seen_airsim16_collection_source.json` is
deployed to the eval H100 annotation path.

## Base

`7929593830954eb02b1bb8b9a16a089d8f0966c7`

## Changes

- Added a PathExpert-only oracle runner for collection-40.
  - Pure gate predicate passes only for `SR >= 0.80`, `median_NE < 20`, and
    `projection_failures == 0`.
  - Writes atomic `oracle_gate.json` output.
  - Runs optional B0 `step_004000.pt` shadow inference on the first 10 routes;
    shadow actions are label-only and never control the bridge.
  - Computes pilot cross-track P95, freezes thresholds through
    `experiments.aerial.takeover.freeze_thresholds`, and writes them with the
    collection-source SHA256 into the collection manifest.
- Added a launcher that waits for
  `/tmp/aerial_eval_cache/logs/eval/b0_seen_videos.status`, exits on `FAILED`,
  and launches DAgger only after `COMPLETED`.
  - Refuses missing collection source, manifest, checkpoint, failed oracle
    gate, or a checkpoint not named `step_004000.pt`.
  - Uses the frozen manifest thresholds and the
    `aerial_joint_b0_novideo` task.
- Added the blocking eval-H100 source deployment checklist. It explicitly
  prohibits deriving a collection source from held-out data.

## TDD Evidence

The oracle tests first failed because `run_oracle_gate` did not exist. The
shadow-control regression then failed because `run_oracle_episode` did not
accept a shadow policy. Both were implemented only after observing those
expected failures.

## Verification

- Focused oracle/takeover/collector tests: `19 passed`.
- Full `experiments/aerial/tests` suite: `71 passed`.
- `bash -n experiments/aerial/scripts/wait_videos_then_collect.sh`: passed.
- Cursor diagnostics: no linter errors in the new Python files.

## Concerns

- Real AirSim/OpenFly execution was not possible on this macOS host.
- The required collection source is intentionally not fabricated or committed;
  remote collection is blocked until its approved artifact is deployed.
- The worktree had pre-existing modified and untracked files at start; they
  were not included in this task's commit.
# Task 5 report: M0 gate — train ≥1 epoch / 50 steps

**Status:** BLOCKED — no NVIDIA GPU on this macOS host; M0 train not executed.

**Branch:** `feat/aerial-wam-phase1`  
**Base:** `591373c`  
**Date:** 2026-07-17

## Deliverables

| Artifact | Status |
|----------|--------|
| `experiments/aerial/README.md` | Done — Linux/CUDA runbook with exact commands |
| `experiments/aerial/m0_preflight.py` | Done — CPU wiring checks (not M0 evidence) |
| 50-step train + checkpoint | **Not done** — blocked on GPU + data + text cache |

## Commits

```
ede2f36 docs(aerial): add Phase-1 M0 training runbook
721a7ad docs(aerial): add M0 preflight helper and macOS notes
```

## Local sanity (macOS, no CUDA)

Environment: `/Users/xudazhong/Projects/FastWAM/.venv/bin/python` (torch 3.10, MPS available, **CUDA false**).

| Check | Result |
|-------|--------|
| `nvidia-smi` | unavailable |
| `pytest experiments/aerial/tests/ -q` | **21 passed** |
| Hydra `train` + `task=aerial_joint_1cam_1e-4` | `action_dim=4`, `cams=1`, `max_steps=50` |
| `verify_aerial_source.py` on smoke | `passed: true` (2 trajs) |
| `m0_preflight.py` on smoke | verify OK; **sample_ok=False** — missing `data/text_embeds_cache/openfly/*.pt` |
| `bash scripts/train_zero2.sh 1 task=...` | hangs on DeepSpeed distributed init (no CUDA) |
| `python scripts/train.py task=...` (no accelerate) | writes `config.yaml`, then loads Wan2.2 weights on CPU — not viable for 50 steps |

Data: `data/openfly_lerobot/train_subset` → symlink to `smoke` (2 episodes, 16 frames). **Insufficient for real M0** (brief requires ≥200 trajs).

## M0 not claimed

No run reached 50 optimizer steps. No `loss_video` / `loss_action` logged. No `step_000050` checkpoint produced.

## Run on Linux CUDA host

Follow `experiments/aerial/README.md`:

1. **Data (≥200 trajs):**
   ```bash
   python -m experiments.aerial.download_openfly_subset \
     --config experiments/aerial/subset_manifest.example.yaml --max-trajs 200
   python -m experiments.aerial.convert_openfly_to_lerobot \
     --ann data/openfly_raw/Annotation/subset_train.json \
     --image-root data/openfly_raw --out data/openfly_lerobot/train_subset
   python experiments/aerial/verify_aerial_source.py \
     --lerobot-root data/openfly_lerobot/train_subset --sample-size 200
   ```

2. **Text cache:**
   ```bash
   torchrun --standalone --nproc_per_node=8 \
     scripts/precompute_text_embeds.py task=aerial_joint_1cam_1e-4
   ```

3. **Train (50 steps):**
   ```bash
   export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
   RUN_ID="m0-$(date +%Y%m%d-%H%M%S)" export RUN_ID
   bash scripts/train_zero2.sh 8 task=aerial_joint_1cam_1e-4 mixed_precision=bf16
   ```

## Evidence to capture when M0 passes

- Log line: `[done] max_steps reached step=50 ...`
- Finite `loss_video=` and `loss_action=` in `[train]` log lines
- `runs/aerial_joint_1cam_1e-4/$RUN_ID/checkpoints/weights/step_000050.pt`
- `runs/aerial_joint_1cam_1e-4/$RUN_ID/checkpoints/state/step_000050/`
- GPU model/count, trajectory count from `meta/info.json`

## Concerns

1. Smoke symlink (2 trajs) must be replaced before GPU M0.
2. Text embed cache is mandatory; dataloader fails without it.
3. `train_zero2.sh` needs venv `accelerate` on PATH and CUDA for DeepSpeed.
4. Wan2.2-5B download/checkpoint path must be configured on the GPU host.
