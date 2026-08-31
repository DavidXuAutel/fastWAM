#!/usr/bin/env bash
# Called on robot: ./remote_viewer_cmd.sh [:0|:1|...]
set -euo pipefail
export DISPLAY="${1:-:1}"
export MUJOCO_GL=glfw
LOG=/tmp/mujoco_genie_viewer.log
: >"$LOG"
nohup bash "$(dirname "$0")/start_genie_g1_viewer.sh" >>"$LOG" 2>&1 &
echo "started viewer (DISPLAY=$DISPLAY) pid=$! log=$LOG"
sleep 3
tail -n 40 "$LOG" || true
