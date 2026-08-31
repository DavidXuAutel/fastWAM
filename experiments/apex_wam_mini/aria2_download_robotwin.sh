#!/usr/bin/env bash
# Generate aria2 batch input list and launch parallel download of all
# RoboTwin2.0 files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="data/official/robotwin2.0_raw"
LIST="/tmp/robotwin_aria2_list.txt"
LIST_PERSIST="data/apex_wam_mini/reports/robotwin_aria2_list.txt"
LOG="data/apex_wam_mini/reports/aria2_robotwin.log"
LOCK="data/apex_wam_mini/reports/aria2_robotwin.lock"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Iseconds)] aria2_robotwin already running (lock held); exit" >&2
  exit 0
fi

mkdir -p "$OUT_DIR" data/apex_wam_mini/reports

echo "=== generate file list ==="
"$PYTHON" - <<'PY'
from huggingface_hub import HfApi

info = HfApi().dataset_info("TianxingChen/RoboTwin2.0")
paths = ["/tmp/robotwin_aria2_list.txt", "data/apex_wam_mini/reports/robotwin_aria2_list.txt"]
for path in paths:
    with open(path, "w") as f:
        for s in info.siblings:
            if s.rfilename in (".gitattributes",):
                continue
            f.write(
                f"https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/{s.rfilename}?download=true\n"
            )
            f.write(f"  out={s.rfilename}\n")
print("files", sum(1 for _ in open(paths[0]) if _.startswith("  out=")))
PY
wc -l "$LIST" "$LIST_PERSIST"

echo "=== stop competing hf_hub download for this repo ==="
pkill -9 -f 'TianxingChen/RoboTwin2.0.*robotwin2.0_raw' 2>/dev/null || true
pkill -9 -f 'download_official_full.sh robotwin_raw' 2>/dev/null || true
sleep 2
rm -f data/official/.download.lock

echo "=== launch aria2 (j=8 parallel files, x=8 conn each, continue partial) ==="
nohup aria2c \
  --input-file="$LIST" \
  --dir="$OUT_DIR" \
  --continue=true \
  --max-concurrent-downloads=8 \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=1M \
  --max-tries=5 \
  --retry-wait=3 \
  --console-log-level=warn \
  --summary-interval=0 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  > "$LOG" 2>&1 &
echo "ARIA2_PID=$!"
sleep 3
echo "=== proc ==="
pgrep -fl aria2c | head
