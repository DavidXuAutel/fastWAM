#!/usr/bin/env bash
tail -n 50 /tmp/mujoco_genie_viewer.log 2>/dev/null || echo "(no log yet)"
echo "--- processes ---"
pgrep -af "launch_genie|mujoco|Simulate" 2>/dev/null || true
