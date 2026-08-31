#!/usr/bin/env bash
# Start ScoutXWAM HTTP serve on H100 (port 8010).
set -euo pipefail
ROOT="${SCOUT_PACKAGE_ROOT:-/home/a25689/FastWAM/scoutxwam_droid100_inference}"
PY="${SCOUT_PYTHON:-/home/a25689/micromamba/envs/mot-wam/bin/python}"
HOST="${SCOUT_HOST:-127.0.0.1}"
PORT="${SCOUT_PORT:-8010}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/serve_$(date +%Y%m%d_%H%M%S).log"

cd "$ROOT"
# Prefer package-local serve.py; fall back to FastWAM experiments copy.
SERVE="$ROOT/serve.py"
if [[ ! -f "$SERVE" ]]; then
  SERVE="/home/a25689/FastWAM/experiments/scoutxwam_franka_bridge/serve.py"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/X-WAM:${PYTHONPATH:-}"

echo "[start_serve] $PY $SERVE --host $HOST --port $PORT  (log=$LOG)"
nohup "$PY" -u "$SERVE" \
  --package-root "$ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  >"$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
sleep 2
tail -n 20 "$LOG" || true
