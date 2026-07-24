# Task 6 report: 75/25 weighted source dataset

**Status:** COMPLETE

## Deliverables

- Added `WeightedSourceDataset` with seeded probabilistic source selection,
  `data_source` labels, and resetting `pop_source_counts()`.
- Added a dedicated aerial Hydra factory that instantiates the original and
  DAgger-correction `RobotVideoDataset` sources without using concat semantics.
- Added FT source monitoring in the trainer: counts log every 50 steps and each
  complete 200-step window must have a correction rate in `[0.20, 0.30]`.
- Added the requested FT data/task configs with 75/25 mixing, action-only loss,
  `1e-5` learning rate, 1000 steps, and 250-step checkpoints.

## TDD and verification

- RED: new tests failed because the weighted dataset/factory/monitor and trainer
  hook did not exist.
- GREEN: `test_ft_weighted_source.py` — 7 passed.
- Regression: full `experiments/aerial/tests` suite — 78 passed.
- Hydra composition resolved successfully for
  `task=aerial_joint_b0_ft_dagger` and all locked values were asserted.
- IDE lint diagnostics reported no errors in changed Python files.

## Concerns

- Source monitoring is FT-config gated and evaluates each distributed process's
  sampled batch stream independently; the fixed seed's five 200-step windows
  contain 48, 57, 51, 55, and 46 corrections, all within the required range.
- Dataset-side counters are process-local with DataLoader workers, so run
  enforcement intentionally uses collated `data_source` labels in the trainer.

## Important-fix follow-up (2026-07-24)

- Set FT `num_workers: 0` so the seeded `WeightedSourceDataset` generator is
  consumed only in the main process and cannot be cloned into correlated worker
  streams.
- Added a deterministic five-window check: seed 42 produces correction counts
  `[48, 57, 51, 55, 46]`, all within the required 40–60 per 200 samples.
- Added `FTSourceMonitor.reset(start_step=...)`, invoked after checkpoint
  restoration. When resume lands inside an absolute 200-step window, that first
  partial window is logged and skipped; enforcement resumes on the next complete
  aligned 200-step window.
- TDD RED reproduced `num_workers: 2`, missing monitor reset behavior, and the
  absent trainer reset hook; GREEN targeted suite: 11 passed.
- Covering regression suite: 82 passed. Hydra composition confirmed
  `num_workers == 0`; changed-file lint diagnostics were clean.
