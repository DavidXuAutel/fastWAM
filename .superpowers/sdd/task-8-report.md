# Task 8 report: Held-out compare and S1 gate

**Status:** COMPLETE (implementation and local validation)
**Branch:** `feat/aerial-wam-phase1`
**Date:** 2026-07-25

## Deliverables

- `compare_finetune.py` compares the locked B0 `step_004000.json` against
  fine-tune checkpoints, selects the lowest mean NE, and returns success only
  when that NE is at most `108.75649833236835`.
- The report includes mean/median NE, SR/SPL, matched per-episode NE deltas,
  improve/flat/regress counts, and quantization L2 statistics when supplied.
- A failed S1 gate writes a diagnosis scaffold with explicit prohibitions
  against automatic data expansion and unseen evaluation.
- `eval_ft_ckpts_seen20.sh` locks checkpoints 250/500/1000, held-out seen-20,
  seed 42, 100 steps, and `aerial_joint_b0_novideo`.

## Commit

`feat(aerial): add FT held-out compare and S1 gate`

## Verification

- RED: compare module import failed before implementation.
- RED: eval script contract failed while the script was absent.
- RED: unlocked baseline test showed the CLI accepted NE `135.0`.
- GREEN: `python3 -m pytest experiments/aerial/tests/test_compare_finetune.py -q`
  — 5 passed.
- `bash -n experiments/aerial/scripts/eval_ft_ckpts_seen20.sh` — passed.
- IDE diagnostics for the Python deliverables — clean.

## Concerns

- No GPU/OpenFly held-out evaluation was run locally; real S1 status remains
  pending production of the three FT metrics files.
- The current closed-loop evaluator writes aggregate SR/NE/SPL only. The
  comparator preserves the required per-episode and quantization analyses when
  those optional records are present, but the evaluator must expose them for a
  populated real report.
- On S1 failure, the workflow stops after report/diagnosis generation; it does
  not expand data or start unseen evaluation.

---

# Task 8 report: Phase-1 checklist (after Tasks 1–7)

**Status:** DONE

**Branch:** `feat/aerial-wam-phase1`
**Base:** `24bfa83`
**Date:** 2026-07-17

## Deliverables

| Artifact | Status |
|----------|--------|
| `experiments/aerial/README.md` | Done — Phase-1 gate checklist table with M0/M1a mock/M1a real |
| `experiments/aerial/eval/README.md` | Done — cross-link to main checklist |

## Gate summary

| Gate | Status | Notes |
|------|--------|-------|
| M0 train (50 steps) | **BLOCKED** | Linux CUDA + ≥200 trajs + text cache required; runbook linked, not executed |
| M1a mock | **DONE** | Mock/replay smoke produces finite SR/NE/SPL JSON (Task 7) |
| M1a real | **PENDING** | Linux OpenFly/AirSim + M0 checkpoint; ≥20 episodes not run |

M0 and M1a real gates are **not claimed passed**.

## Commits

```
d2ca474 docs(aerial): add Phase-1 gate checklist after Tasks 1–7
```

## Evidence links

- **M0:** `experiments/aerial/README.md` — dataset prep, text embeds, `train_zero2.sh`, checkpoint paths
- **M1a mock:** `experiments/aerial/eval/README.md#evidence-mock-smoke`
- **M1a real:** `experiments/aerial/eval/README.md#linux-clone-openfly-platform` + CLI section

## Out of scope (Phase 2+)

- M1b / M2, B1 > B0 ablation, TT bridge, Urban Canyon — see robomaster-tt-control plan docs.
