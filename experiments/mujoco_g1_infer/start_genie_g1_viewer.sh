#!/usr/bin/env bash
# Run from ~/mujoco_g1_infer after sync. Needs DISPLAY (desktop or ssh -X/-Y).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
exec python3 "$ROOT/launch_genie_g1_viewer.py" "$@"
