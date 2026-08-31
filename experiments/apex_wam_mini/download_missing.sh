#!/usr/bin/env bash
# Fill gaps: RoboTwin missing zips + verify LIBERO completeness + finalize marker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
LOG="data/apex_wam_mini/reports/official_full_download.log"
MISSING_LOG="data/apex_wam_mini/reports/download_missing.log"
OFFICIAL="data/official"
DONE_MARKER="${OFFICIAL}/.ALL_DONE"
ROBOTWIN_OUT="${OFFICIAL}/robotwin2.0_raw"
ARIA2_LIST="${ROOT}/data/apex_wam_mini/reports/robotwin_aria2_list.txt"

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p data/apex_wam_mini/reports

mlog() { echo "[$(date -Iseconds)] $*" | tee -a "$MISSING_LOG"; }

ensure_hf() {
  "$PYTHON" - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23"])
print("huggingface_hub ok")
PY
}

stop_competing_robotwin() {
  pkill -9 -f '[T]ianxingChen/RoboTwin2.0.*robotwin2.0_raw' 2>/dev/null || true
  pkill -9 -f '[d]ownload_official_full.sh robotwin_raw' 2>/dev/null || true
  sleep 2
  rm -f "${OFFICIAL}/.download.lock"
}

download_robotwin_missing() {
  mlog "RoboTwin: download missing zips"
  stop_competing_robotwin
  "$PYTHON" - "$ROBOTWIN_OUT" "$ARIA2_LIST" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

out_dir = Path(sys.argv[1])
list_path = Path(sys.argv[2])
repo = "TianxingChen/RoboTwin2.0"

missing = []
if list_path.exists():
    for line in list_path.read_text().splitlines():
        if line.startswith("  out="):
            rel = line.split("=", 1)[1]
            target = out_dir / rel
            if not target.is_file() or target.stat().st_size == 0:
                missing.append(rel)
else:
    missing = [
        "dataset/beat_block_hammer/piper_randomized_500.zip",
        "dataset/blocks_ranking_rgb/aloha-agilex_randomized_500.zip",
        "dataset/blocks_ranking_rgb/arx-x5_randomized_500.zip",
    ]

print(f"missing_count={len(missing)}")
for rel in missing:
    partial = out_dir / rel
    partial.parent.mkdir(parents=True, exist_ok=True)
    aria2_partial = Path(str(partial) + ".aria2")
    if aria2_partial.exists():
        aria2_partial.unlink()
    print(f"downloading {rel} ...")
    hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=rel,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    size = (out_dir / rel).stat().st_size
    print(f"done {rel} size={size}")
PY
  mlog "RoboTwin missing zips complete"
}

verify_libero_dataset() {
  local out="$1" label="$2"
  mlog "${label}: verify LeRobot completeness (parquet images or videos/)"
  "$PYTHON" - "$out" "$label" <<'PY'
import json
import sys
from pathlib import Path

out, label = Path(sys.argv[1]), sys.argv[2]
info_path = out / "meta/info.json"
if not info_path.exists():
    raise SystemExit(f"{label}: missing meta/info.json")

info = json.loads(info_path.read_text())
parquets = sorted((out / "data").rglob("*.parquet"))
if not parquets:
    raise SystemExit(f"{label}: no parquet under data/")

videos = list((out / "videos").rglob("*.mp4")) if (out / "videos").exists() else []
if videos:
    print(f"{label}: ok videos={len(videos)} episodes={info.get('total_episodes')}")
    sys.exit(0)

# Official LIBERO repos store frames as image bytes inside parquet (no videos/ on HF).
try:
    import pyarrow.parquet as pq
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
    import pyarrow.parquet as pq

table = pq.read_table(parquets[0])
image_cols = [c for c in table.column_names if "image" in c.lower()]
if not image_cols:
    raise SystemExit(f"{label}: no image columns in {parquets[0]}")

col = image_cols[0]
cell = table[col][0].as_py()
if isinstance(cell, dict) and cell.get("bytes"):
    nbytes = len(cell["bytes"])
    print(f"{label}: ok parquet_embedded_images bytes={nbytes} episodes={info.get('total_episodes')}")
elif isinstance(cell, (bytes, bytearray)) and len(cell) > 0:
    print(f"{label}: ok parquet_raw_bytes episodes={info.get('total_episodes')}")
else:
    raise SystemExit(f"{label}: image column present but empty in {parquets[0]}")
PY
  mlog "${label}: verify complete"
}

download_libero_videos() {
  local repo="$1" out="$2" label="$3"
  if "$PYTHON" - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi

repo = sys.argv[1]
files = HfApi().list_repo_files(repo, repo_type="dataset")
has_videos = any(f.startswith("videos/") and f.endswith(".mp4") for f in files)
print("1" if has_videos else "0")
PY
  then
    mlog "${label}: repo has videos/ on HF, snapshot videos/**"
    "$PYTHON" - "$repo" "$out" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, out = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=out,
    allow_patterns=["videos/**"],
    max_workers=16,
)
print(f"done {repo} videos -> {out}")
PY
  else
    mlog "${label}: no videos/ on HF; frames are parquet-embedded — verify only"
    verify_libero_dataset "$out" "$label"
  fi
}

mark_official_complete() {
  stop_competing_robotwin
  if ! grep -q "done TianxingChen/RoboTwin2.0 ->" "$LOG" 2>/dev/null; then
    echo "done TianxingChen/RoboTwin2.0 -> ${ROBOTWIN_OUT} (aria2)" >> "$LOG"
    mlog "marked RoboTwin done in official_full_download.log"
  fi
  date -Iseconds > "$DONE_MARKER"
  mlog "wrote ${DONE_MARKER}"
}

main() {
  local target="${1:-all}"
  ensure_hf
  mlog "download_missing start target=${target}"

  case "$target" in
    robotwin)
      download_robotwin_missing
      ;;
    libero_hfvla)
      download_libero_videos "HuggingFaceVLA/libero" "${OFFICIAL}/libero_hfvla_v2.1" "libero_hfvla"
      ;;
    libero_raw)
      download_libero_videos "physical-intelligence/libero" "${OFFICIAL}/libero_raw" "libero_raw"
      ;;
    finalize)
      verify_libero_dataset "${OFFICIAL}/libero_hfvla_v2.1" "libero_hfvla"
      verify_libero_dataset "${OFFICIAL}/libero_raw" "libero_raw"
      download_robotwin_missing
      mark_official_complete
      ;;
    all)
      download_robotwin_missing
      download_libero_videos "HuggingFaceVLA/libero" "${OFFICIAL}/libero_hfvla_v2.1" "libero_hfvla"
      download_libero_videos "physical-intelligence/libero" "${OFFICIAL}/libero_raw" "libero_raw"
      mark_official_complete
      ;;
    *)
      echo "Usage: $(basename "$0") [robotwin|libero_hfvla|libero_raw|finalize|all]"
      exit 1
      ;;
  esac

  mlog "download_missing ALL DONE: ${target}"
}

main "$@"
