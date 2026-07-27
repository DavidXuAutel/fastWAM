#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
AERIAL_FT_CACHE="${AERIAL_FT_CACHE:-/tmp/aerial_ft_cache}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$AERIAL_FT_CACHE/runs/b1-${STAMP}/checkpoints/weights}"
SHARED_WEIGHTS_DIR="${SHARED_WEIGHTS_DIR:-/home/a25689/aerial_cache_shared/runs/aerial_b1_ft/m1b-${STAMP}/checkpoints/weights}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-/tmp/aerial_eval_cache/orchestration/eval_queue}"
RESULTS_ROOT="${RESULTS_ROOT:-/tmp/aerial_eval_cache/results}"
ANN="${ANN:-/tmp/aerial_eval_cache/Annotation/seen_airsim16_m1a20.json}"
OPENFLY_ROOT="${OPENFLY_ROOT:-/tmp/aerial_eval_cache/OpenFly-Platform}"
TASK="${TASK:-aerial_joint_b1_joint}"
POLL_S="${POLL_S:-60}"
MIN_BYTES="${MIN_BYTES:-1000000000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ONCE=0
DRY_RUN=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --once) ONCE=1 ;;
  -h|--help)
    echo "Usage: STAMP=... $0 [--dry-run|--once]"
    echo "Polls B1 weights and enqueues step_000250/000500/001000 evals without blocking train."
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

command=(
  "$PYTHON_BIN" -m experiments.aerial.orchestration.b1_discover
  --stamp "$STAMP"
  --weights-dir "$WEIGHTS_DIR"
  --queue-dir "$EVAL_QUEUE_DIR"
  --results-root "$RESULTS_ROOT"
  --ann "$ANN"
  --openfly-root "$OPENFLY_ROOT"
  --task "$TASK"
  --steps "250,500,1000"
  --poll-s "$POLL_S"
  --min-bytes "$MIN_BYTES"
)

if (( DRY_RUN )); then
  cat <<EOF
watch B1 checkpoints under:
  WEIGHTS_DIR=$WEIGHTS_DIR
  SHARED_WEIGHTS_DIR=$SHARED_WEIGHTS_DIR
enqueue steps: step_000250 step_000500 step_001000
queue: $EVAL_QUEUE_DIR
poll_s=$POLL_S
EOF
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd "$REPO_ROOT"
if (( ONCE )); then
  PYTHONPATH=. "${command[@]}" --once
else
  PYTHONPATH=. "${command[@]}"
fi
