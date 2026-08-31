#!/usr/bin/env bash
# Download Franka-profile open datasets (LIBERO extensions + DROID + small Franka sets).
# Uses adaptive parallelism: snapshot_download (scaled workers) or aria2 for many files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
FETCH="${ROOT}/experiments/apex_wam_mini/hf_parallel_fetch.py"
FRANKA_DIR="${ROOT}/data/franka"
LOG="${ROOT}/data/apex_wam_mini/reports/franka_download.log"
LOCK="${ROOT}/data/apex_wam_mini/reports/franka_download.lock"
DONE_MARKER="${FRANKA_DIR}/.ALL_DONE"
REPORTS="${ROOT}/data/apex_wam_mini/reports"

# >=150 files -> aria2 batch; else snapshot with up to 32 workers
export HF_PARALLEL_FILE_THRESHOLD="${HF_PARALLEL_FILE_THRESHOLD:-150}"
export HF_SNAPSHOT_MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-32}"
export ARIA2_MAX_CONCURRENT="${ARIA2_MAX_CONCURRENT:-16}"
export ARIA2_CONNECTIONS_PER_SERVER="${ARIA2_CONNECTIONS_PER_SERVER:-8}"

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=120

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Iseconds)] franka download already running; exit" >&2
  exit 0
fi

mkdir -p "$FRANKA_DIR" "$REPORTS"

mlog() { echo "[$(date -Iseconds)] $*"; }

ensure_deps() {
  "$PYTHON" - <<'PY'
import importlib.util, subprocess, sys
for pkg in ("huggingface_hub",):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23"])
print("huggingface_hub ok")
PY
  if ! command -v aria2c >/dev/null 2>&1; then
    mlog "WARN aria2c missing; large repos will use snapshot_download (slower)"
  else
    mlog "aria2c available: $(command -v aria2c)"
  fi
}

link_libero_90() {
  local src="${ROOT}/data/official/libero_nvidia_lerobot_v3/libero_90"
  local dst="${FRANKA_DIR}/libero_90"
  if [[ -f "${src}/meta/info.json" ]]; then
    ln -sfn "$src" "$dst"
    mlog "linked libero_90 -> ${src}"
  else
    mlog "WARN libero_90 official missing; fetching nvidia subdir"
    fetch_dataset "nvidia/LIBERO_LeRobot_v3" "$dst" "libero_90/**"
  fi
}

fetch_dataset() {
  local repo="$1" out="$2" patterns="${3:-}"
  local slug
  slug="$(basename "$out")"
  mlog "fetch ${repo} -> ${out} patterns=${patterns:-ALL}"
  "$PYTHON" "$FETCH" \
    --repo "$repo" \
    --out "$out" \
    ${patterns:+--patterns "$patterns"} \
    --skip-if-info \
    --aria2-list "${REPORTS}/aria2_${slug}_list.txt" \
    --aria2-log "${REPORTS}/aria2_${slug}.log" \
    >> "$LOG" 2>&1
  mlog "complete ${repo} -> ${out}"
}

main() {
  local target="${1:-all}"
  ensure_deps
  mlog "franka download start target=${target} threshold=${HF_PARALLEL_FILE_THRESHOLD}"

  case "$target" in
    libero_90) link_libero_90 ;;
    lerobot_libero) fetch_dataset "lerobot/libero" "${FRANKA_DIR}/lerobot_libero" ;;
    libero_plus) fetch_dataset "lerobot/libero_plus" "${FRANKA_DIR}/libero_plus" ;;
    libero_long) fetch_dataset "robbyant/libero-long-lerobot" "${FRANKA_DIR}/libero_long" ;;
    droid_lerobot) fetch_dataset "IPEC-COMMUNITY/droid_lerobot" "${FRANKA_DIR}/droid_lerobot" ;;
    droid_1.0.1) fetch_dataset "lerobot/droid_1.0.1" "${FRANKA_DIR}/droid_1.0.1" ;;
    franka_move) fetch_dataset "Topasm/Franka_move" "${FRANKA_DIR}/franka_move" ;;
    cmu_franka) fetch_dataset "lerobot/cmu_franka_exploration_dataset" "${FRANKA_DIR}/cmu_franka_exploration" ;;
    all)
      link_libero_90
      fetch_dataset "lerobot/libero" "${FRANKA_DIR}/lerobot_libero"
      fetch_dataset "lerobot/libero_plus" "${FRANKA_DIR}/libero_plus"
      fetch_dataset "robbyant/libero-long-lerobot" "${FRANKA_DIR}/libero_long"
      fetch_dataset "Topasm/Franka_move" "${FRANKA_DIR}/franka_move"
      fetch_dataset "lerobot/cmu_franka_exploration_dataset" "${FRANKA_DIR}/cmu_franka_exploration"
      fetch_dataset "IPEC-COMMUNITY/droid_lerobot" "${FRANKA_DIR}/droid_lerobot"
      fetch_dataset "lerobot/droid_1.0.1" "${FRANKA_DIR}/droid_1.0.1"
      ;;
    *)
      echo "Usage: $(basename "$0") [libero_90|lerobot_libero|libero_plus|libero_long|droid_lerobot|droid_1.0.1|franka_move|cmu_franka|all]"
      exit 1
      ;;
  esac

  date -Iseconds > "$DONE_MARKER"
  mlog "franka download ALL DONE: ${target}"
  du -sh "${FRANKA_DIR}"/* 2>/dev/null | tee -a "$LOG" || true
}

main "$@"
