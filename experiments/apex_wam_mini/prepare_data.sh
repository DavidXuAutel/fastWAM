#!/usr/bin/env bash
# Scan local datasets and build Apex-WAM-Mini stage manifests.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python3 experiments/apex_wam_mini/prepare_data.py all "$@"
