#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
OPENFLY_ROOT="${OPENFLY_ROOT:-}"
HELDOUT_ANN="${HELDOUT_ANN:-}"
B0_METRICS="${B0_METRICS:-}"
AERIAL_FT_CACHE="${AERIAL_FT_CACHE:-/tmp/aerial_ft_cache}"
FT_RUN_DIR="${FT_RUN_DIR:-}"
RESULT_DIR="${RESULT_DIR:-$AERIAL_FT_CACHE/results/heldout-seen20}"
TASK="${TASK:-aerial_joint_b0_novideo}"
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    echo "Usage: OPENFLY_ROOT=... HELDOUT_ANN=... B0_METRICS=... $0 [--dry-run]"
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

if [[ -z "$FT_RUN_DIR" && -f "$AERIAL_FT_CACHE/ft.run_dir" ]]; then
  FT_RUN_DIR="$(tr -d '\r\n' < "$AERIAL_FT_CACHE/ft.run_dir")"
fi

if (( ! DRY_RUN )); then
  [[ -d "$REPO_DIR" ]] || { echo "Missing repository: $REPO_DIR" >&2; exit 2; }
  [[ -d "$OPENFLY_ROOT" ]] || { echo "Missing OPENFLY_ROOT: $OPENFLY_ROOT" >&2; exit 2; }
  [[ -f "$HELDOUT_ANN" ]] || { echo "Missing held-out annotation: $HELDOUT_ANN" >&2; exit 2; }
  [[ -f "$B0_METRICS" ]] || { echo "Missing B0 metrics: $B0_METRICS" >&2; exit 2; }
  [[ "$(basename "$B0_METRICS")" == "step_004000.json" ]] || {
    echo "Refusing unlocked baseline metrics: expected step_004000.json" >&2
    exit 2
  }
  [[ -n "$FT_RUN_DIR" && -d "$FT_RUN_DIR" ]] || {
    echo "Missing FT_RUN_DIR (or $AERIAL_FT_CACHE/ft.run_dir)" >&2
    exit 2
  }
  mkdir -p "$RESULT_DIR"
fi

candidate_args=()
for step in 250 500 1000; do
  checkpoint="$(printf '%s/checkpoints/weights/step_%06d.pt' "$FT_RUN_DIR" "$step")"
  metrics="$(printf '%s/step_%06d.json' "$RESULT_DIR" "$step")"
  command=(
    python3 -m experiments.aerial.eval.run_closed_loop
    --bridge openfly
    --policy fastwam
    --openfly-root "$OPENFLY_ROOT"
    --ann "$HELDOUT_ANN"
    --checkpoint "$checkpoint"
    --out "$metrics"
    --max-episodes 20
    --max-steps 100
    --seed 42
    --task "$TASK"
  )
  if (( DRY_RUN )); then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    [[ -f "$checkpoint" ]] || { echo "Missing FT checkpoint: $checkpoint" >&2; exit 2; }
    (
      cd "$REPO_DIR"
      PYTHONPATH=. "${command[@]}"
    )
  fi
  candidate_args+=(--candidate "$step=$metrics")
done

report="$RESULT_DIR/ft_selection_report.json"
diagnosis="$RESULT_DIR/ft_s1_failure_diagnosis.json"
compare_command=(
  python3 -m experiments.aerial.eval.compare_finetune
  --baseline "$B0_METRICS"
  "${candidate_args[@]}"
  --out "$report"
  --diagnosis-out "$diagnosis"
)
if (( DRY_RUN )); then
  printf '%q ' "${compare_command[@]}"
  printf '\n'
else
  (
    cd "$REPO_DIR"
    PYTHONPATH=. "${compare_command[@]}"
  )
fi
