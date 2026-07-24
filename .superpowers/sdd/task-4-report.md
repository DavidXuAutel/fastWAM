# Task 4 Report: DAgger Collector and Correction LeRobot Writer

## Status

Implemented the DAgger collection loop, correction JSONL/image persistence, atomic
collection manifest updates, LeRobot v2.1 correction conversion, and the
expert-to-policy counter reset needed to prevent immediate takeover oscillation.

## Changes

- Added `experiments/aerial/eval/collect_dagger.py`.
  - Labels every captured state with the continuous 4D `PathExpert` action.
  - Executes expert actions during intervention and policy actions otherwise.
  - Prefers `bridge.step_delta`; falls back to the nearest primitive only for
    bridges without continuous stepping.
  - Writes episode JSONL atomically alongside RGB PNG frames.
  - Supports configurable abort-tail truncation.
  - Updates `manifest.json` atomically after each episode with completed,
    failed, and frozen threshold data.
- Added `experiments/aerial/write_correction_lerobot.py`.
  - Loads collected JSONL and images into the existing aerial LeRobot schema.
  - Validates 4D state/action shapes, finite values, image existence, and task
    text before delegating to the existing LeRobot v2 writer.
  - Uses FPS 10 and `meta.action_source=pos_delta_v1` through the existing
    converter constants/writer.
  - Defaults output to `data/openfly_lerobot/b0_dagger_correction`.
- Patched `TakeoverController` to clear stall and worsen counters when expert
  control releases back to policy.
- Added focused collector, writer, and takeover regression tests.

## TDD Evidence

Before implementation, the new collector and writer tests failed during
collection because their modules did not exist. The takeover regression test
failed with `expert != policy`, reproducing the stale-stall-counter oscillation.
After implementation, all focused tests passed.

## Verification

- `uv run --no-project --with pytest python3 -m pytest experiments/aerial/tests/test_collect_dagger_mock.py experiments/aerial/tests/test_write_correction_lerobot.py experiments/aerial/tests/test_takeover.py -q`
  - 11 passed.
- `uv run --no-project --with pytest python3 -m pytest experiments/aerial/tests -q`
  - 60 passed.
- Cursor diagnostics reported no linter errors in changed files.

## Concerns

The worktree contained substantial pre-existing uncommitted changes before Task
4, including the continuous `step_delta` and pose-normalization helpers consumed
here. Those files were not modified or staged as part of this task.

## Reviewer Fixes

- All policy/expert control-mode transitions now reset worsen, stall,
  release-stability, and no-progress counters together.
- Added a regression proving a stall-triggered takeover and release cannot
  consume leftover no-progress budget and abort on the next policy step.
- Bridge and policy construction now occur inside the per-episode guarded
  lifecycle. Bridge and policy setup failures become atomic failed manifest
  entries, and a bridge created before policy failure is closed.
- Added explicit coverage showing the persisted training `action` remains the
  continuous expert label when the executed policy action differs.

### Fix Evidence

- Red: the three focused regressions initially failed with an early
  `no_progress_abort`, an uncaught policy setup exception, and an uncaught
  bridge setup exception.
- Green: takeover and collector suites passed, 12 tests.
- Full aerial suite: 64 tests passed.
- Cursor diagnostics reported no linter errors in the four changed code/test
  files.
