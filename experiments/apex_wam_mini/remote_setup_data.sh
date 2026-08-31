#!/usr/bin/env bash
# Remote one-shot setup for Apex-WAM-Mini data prep on GPU/CPU server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/opt/conda/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

echo "== FastWAM root: $ROOT =="
"$PYTHON" --version

echo "== Directory layout =="
mkdir -p data/apex_wam_mini/target_robot_teleop
mkdir -p data/apex_wam_mini/mujoco_g1_proxy
mkdir -p data/apex_wam_mini/failure_rollout
mkdir -p data/apex_wam_mini/manifests
mkdir -p data/apex_wam_mini/reports/registry_verification
mkdir -p data/libero_mujoco3.3.2
mkdir -p data/robotwin2.0

echo "== Scan & manifest =="
"$PYTHON" experiments/apex_wam_mini/prepare_data.py all

echo "== Done. Next: bash experiments/apex_wam_mini/download_video_only_datasets.sh libero =="
