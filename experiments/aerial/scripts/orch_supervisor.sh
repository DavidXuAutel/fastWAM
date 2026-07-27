#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${STAMP:-20260727-072347-5k-2gpu-b0-to-joint-video}"
STATUS_PATH="${STATUS_PATH:-/tmp/aerial_cache/orchestration/status.json}"
EVAL_QUEUE_DIR="${EVAL_QUEUE_DIR:-/tmp/aerial_eval_cache/orchestration/eval_queue}"
ORCH_ROOT="${ORCH_ROOT:-/tmp/aerial_cache/orchestration}"
RESULTS_ROOT="${RESULTS_ROOT:-/tmp/aerial_eval_cache/results}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/home/a25689/aerial_cache_shared/runs/aerial_joint_b0_to_joint_video/m1b-${STAMP}/checkpoints/weights}"
CANDIDATE_STEPS="${CANDIDATE_STEPS:-1000,2000,3000,4000,5000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POLL_S="${POLL_S:-30}"
DRY_RUN=0
ONCE=0

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --once) ONCE=1 ;;
  -h|--help)
    echo "Usage: STAMP=... $0 [--dry-run|--once]"
    echo "Advances EVAL_B0_CHECKPOINTS → LOCK_BASELINE → B1_GATES (then orch_b1_gates.sh)."
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

supervisor() {
  PYTHONPATH=. "$PYTHON_BIN" -m experiments.aerial.orchestration.supervisor "$@"
}

run_once() {
  local phase
  if [[ ! -f "$STATUS_PATH" ]]; then
    supervisor --status "$STATUS_PATH" --stamp "$STAMP" --init
  fi
  phase="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("phase",""))' "$STATUS_PATH")"

  if [[ "$phase" == "EVAL_B0_CHECKPOINTS" ]]; then
    supervisor \
      --status "$STATUS_PATH" \
      --stamp "$STAMP" \
      --advance-from-eval-queue \
      --queue-dir "$EVAL_QUEUE_DIR"
    phase="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$STATUS_PATH")"
  fi

  if [[ "$phase" == "LOCK_BASELINE" ]]; then
    supervisor \
      --status "$STATUS_PATH" \
      --stamp "$STAMP" \
      --lock-baseline \
      --weights-dir "$WEIGHTS_DIR" \
      --results-root "$RESULTS_ROOT" \
      --steps "$CANDIDATE_STEPS" \
      --out "$ORCH_ROOT/baseline_lock.manifest.json" || true
    phase="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$STATUS_PATH")"
  fi

  if [[ "$phase" == "B1_GATES" ]]; then
    "$SCRIPT_DIR/orch_b1_gates.sh" || true
  fi

  phase="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("phase",""))' "$STATUS_PATH")"
  echo "$phase"
}

if (( DRY_RUN )); then
  cat <<EOF
supervisor --init/--advance-from-eval-queue → lock_baseline → orch_b1_gates.sh
STATUS_PATH=$STATUS_PATH
EVAL_QUEUE_DIR=$EVAL_QUEUE_DIR
WEIGHTS_DIR=$WEIGHTS_DIR
EOF
  exit 0
fi

cd "$REPO_ROOT"
mkdir -p "$ORCH_ROOT"

if (( ONCE )); then
  run_once
  exit 0
fi

while true; do
  phase="$(run_once)"
  case "$phase" in
    RUN_B1_TRAIN|BLOCKED|FAILED|DONE|S1_REPORT)
      echo "supervisor stopped at phase=$phase"
      exit 0
      ;;
  esac
  sleep "$POLL_S"
done
