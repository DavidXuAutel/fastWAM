#!/usr/bin/env bash
# Download OFFICIAL FULL LIBERO + RoboTwin datasets to remote.
# Not experiment-specific — pull complete official releases.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOCK="${ROOT}/data/official/.download.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Iseconds)] another download_official_full is running (lock held); exit" >&2
  exit 0
fi

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
export PYTHON
# Enable high-performance Xet transfer (anonymous, multi-threaded).
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
# Legacy flag (no-op on new hf_hub but harmless for older versions).
export HF_HUB_ENABLE_HF_TRANSFER=1

OFFICIAL_DIR="${ROOT}/data/official"
mkdir -p "$OFFICIAL_DIR"

log() { echo "[$(date -Iseconds)] $*"; }

ensure_hf() {
  "$PYTHON" - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23"])
print("huggingface_hub ok")
PY
}

snapshot() {
  local repo="$1" out="$2" patterns="${3:-}"
  mkdir -p "$out"
  if [[ -n "$patterns" ]]; then
    "$PYTHON" - "$repo" "$out" "$patterns" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, out, patterns = sys.argv[1], sys.argv[2], sys.argv[3].split('|')
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out,
                  allow_patterns=patterns, max_workers=16)
print(f"done {repo} -> {out}")
PY
  else
    "$PYTHON" - "$repo" "$out" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out, max_workers=16)
print(f"done {repo} -> {out}")
PY
  fi
}

# ---------- RoboTwin ----------
download_robotwin_raw() {
  log "RoboTwin official RAW: TianxingChen/RoboTwin2.0 (~1.47TB)"
  snapshot "TianxingChen/RoboTwin2.0" "${OFFICIAL_DIR}/robotwin2.0_raw"
}

download_robotwin_lerobot() {
  log "RoboTwin LeRobot v3: lerobot/robotwin_unified (~79.6GB)"
  snapshot "lerobot/robotwin_unified" "${OFFICIAL_DIR}/robotwin_unified_lerobot"
}

# ---------- LIBERO ----------
download_libero_raw() {
  log "LIBERO official RAW: physical-intelligence/libero"
  snapshot "physical-intelligence/libero" "${OFFICIAL_DIR}/libero_raw"
}

download_libero_nvidia_v3() {
  log "LIBERO NVIDIA LeRobot v3: nvidia/LIBERO_LeRobot_v3 (4 suites)"
  snapshot "nvidia/LIBERO_LeRobot_v3" "${OFFICIAL_DIR}/libero_nvidia_lerobot_v3"
}

download_libero_hfvla() {
  log "LIBERO HuggingFaceVLA v2.1: HuggingFaceVLA/libero (~35GB)"
  snapshot "HuggingFaceVLA/libero" "${OFFICIAL_DIR}/libero_hfvla_v2.1"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [target]
  target:
    robotwin_raw        TianxingChen/RoboTwin2.0 raw (~1.47TB)
    robotwin_lerobot    lerobot/robotwin_unified v3 (~80GB)
    libero_raw          physical-intelligence/libero
    libero_nvidia       nvidia/LIBERO_LeRobot_v3
    libero_hfvla        HuggingFaceVLA/libero v2.1 (~35GB)
    robotwin            raw + lerobot
    libero              raw + nvidia + hfvla
    all                 everything (default)
EOF
}

main() {
  local target="${1:-all}"
  ensure_hf
  case "$target" in
    robotwin_raw)     download_robotwin_raw ;;
    robotwin_lerobot) download_robotwin_lerobot ;;
    libero_raw)       download_libero_raw ;;
    libero_nvidia)    download_libero_nvidia_v3 ;;
    libero_hfvla)     download_libero_hfvla ;;
    robotwin)         download_robotwin_lerobot; download_robotwin_raw ;;
    libero)           download_libero_hfvla; download_libero_nvidia_v3; download_libero_raw ;;
    all)
      download_libero_hfvla
      download_libero_nvidia_v3
      download_libero_raw
      download_robotwin_lerobot
      download_robotwin_raw
      ;;
    *) usage; exit 1 ;;
  esac
  log "ALL DONE: $target"
}

main "$@"
