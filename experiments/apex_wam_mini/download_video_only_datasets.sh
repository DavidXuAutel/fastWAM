#!/usr/bin/env bash
# Download video-only LeRobot datasets for Apex-WAM-Mini (Stage B).
# Sources: FastWAM README — yuanty/LIBERO-fastwam, yuanty/robotwin2.0-fastwam
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
export PYTHON

LIBERO_DIR="${ROOT}/data/libero_mujoco3.3.2"
ROBOTWIN_DIR="${ROOT}/data/robotwin2.0"
HF_LIBERO="yuanty/LIBERO-fastwam"
HF_ROBOTWIN="yuanty/robotwin2.0-fastwam"

log() { echo "[$(date -Iseconds)] $*"; }

ensure_hf() {
  "$PYTHON" - <<'PY'
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23"])
print("huggingface_hub ok")
PY
}

download_libero() {
  mkdir -p "$LIBERO_DIR"
  log "Downloading LIBERO archives from ${HF_LIBERO} -> ${LIBERO_DIR}"
  "$PYTHON" - <<PY
from huggingface_hub import hf_hub_download, list_repo_files
import os, tarfile, shutil
repo = "${HF_LIBERO}"
out_dir = "${LIBERO_DIR}"
files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".tar.gz")]
if not files:
    raise SystemExit("No .tar.gz files found in repo")
for name in sorted(files):
    dest = os.path.join(out_dir, os.path.basename(name))
    if os.path.exists(dest):
        print(f"skip existing {dest}")
        continue
    print(f"download {name} ...")
    path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", local_dir=out_dir)
    print(f"  -> {path}")
for name in sorted(files):
    archive = os.path.join(out_dir, os.path.basename(name))
    marker = archive + ".extracted"
    if os.path.exists(marker):
        print(f"skip extract {archive}")
        continue
    print(f"extract {archive} ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(out_dir)
    open(marker, "w").close()
print("LIBERO done")
PY
}

download_robotwin() {
  mkdir -p "$ROBOTWIN_DIR"
  log "Downloading RoboTwin split archives from ${HF_ROBOTWIN} -> ${ROBOTWIN_DIR}"
  "$PYTHON" - <<PY
from huggingface_hub import hf_hub_download, list_repo_files
import os, subprocess, glob
repo = "${HF_ROBOTWIN}"
out_dir = "${ROBOTWIN_DIR}"
parts = sorted([f for f in list_repo_files(repo, repo_type="dataset") if "part-" in f or f.endswith(".tar.gz")])
if not parts:
    raise SystemExit("No RoboTwin archive parts found")
for name in parts:
    dest = os.path.join(out_dir, os.path.basename(name))
    if os.path.exists(dest):
        print(f"skip existing {dest}")
        continue
    print(f"download {name} ...")
    hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", local_dir=out_dir)
marker = os.path.join(out_dir, ".robotwin_extracted")
if os.path.exists(marker):
    print("RoboTwin already extracted")
else:
    part_glob = os.path.join(out_dir, "robotwin2.0.tar.gz.part-*")
    parts_local = sorted(glob.glob(part_glob))
    if not parts_local:
        single = os.path.join(out_dir, "robotwin2.0.tar.gz")
        if os.path.exists(single):
            parts_local = [single]
    if not parts_local:
        raise SystemExit(f"No parts under {out_dir}")
    print("concat and extract RoboTwin ...")
    if len(parts_local) > 1:
        subprocess.check_call(f"cat {' '.join(parts_local)} | tar -xzf - -C {out_dir}", shell=True)
    else:
        subprocess.check_call(["tar", "-xzf", parts_local[0], "-C", out_dir])
    open(marker, "w").close()
print("RoboTwin done")
PY
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [libero|robotwin|all]
  libero   — download & extract LIBERO LeRobot (video-only, ~tens of GB)
  robotwin — download & extract RoboTwin (video-only, ~100GB+)
  all      — both
EOF
}

main() {
  local target="${1:-libero}"
  ensure_hf
  case "$target" in
    libero) download_libero ;;
    robotwin) download_robotwin ;;
    all) download_libero; download_robotwin ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
