#!/usr/bin/env bash
echo "=== /tmp/isaac_headless.log (last 60 lines) ==="
tail -n 60 /tmp/isaac_headless.log 2>/dev/null || echo "(no log)"
echo "=== docker ps ==="
docker ps 2>/dev/null | head -12 || echo "docker not available"
echo "=== pgrep isaac / mujoco_g1_infer ==="
pgrep -af "isaac-sim|mujoco_g1_infer" 2>/dev/null || true
