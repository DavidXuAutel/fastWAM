# Task 7 report: Dual-4090 B0 fine-tune workflow

**Status:** COMPLETE (scripts and local validation only)
**Branch:** `feat/aerial-wam-phase1`
**Date:** 2026-07-24

## Deliverables

- `sync_b0_ft_to_4090.sh` packages the committed source tree, records its commit,
  syncs all B0 FT assets to `a25689@10.239.121.14:30879`, and atomically
  installs the cache only after remote SHA256 verification.
- `accelerate_zero2_opt_offload_2proc.yaml` locks two processes, bf16,
  DeepSpeed ZeRO-2, and CPU optimizer offload.
- `smoke_b0_ft_4090.sh` verifies the manifest, loads only
  `step_004000.pt`, runs 1 then 10 steps, checks finite losses and peak memory
  `<23552 MiB/GPU`, and permits exactly one OOM retry with 50M buckets.
- `run_b0_ft_4090.sh` requires a completed smoke, writes
  `ft.status`, restarts from base model weights only, trains 1,000 steps, and
  requires checkpoints at steps 250, 500, and 1,000.
- `test_b0_ft_4090_scripts.py` exercises syntax and no-network dry-run
  contracts.

## Commit

`feat(aerial): add dual-4090 B0 FT sync and smoke runners`

## Verification

- RED before implementation: 4 expected missing-artifact failures.
- `python3 -m pytest experiments/aerial/tests/test_b0_ft_4090_scripts.py -q`
  — 5 passed.
- `bash -n` over all three scripts — passed.
- IDE diagnostics for the new Python test — clean.
- Full aerial suite attempted but collection stopped because system Python has
  no `torch`; no project virtualenv is present in this checkout.

## Concerns

- Per task instructions, no SSH, rsync, GPU smoke, or remote training was run.
- The real acceptance gates (finite GPU losses, `<23 GiB/GPU`, and checkpoint
  production) remain pending execution on the dual-4090 host.
- A second OOM stops immediately and requests ZeRO-3 review; it never retries
  indefinitely.
