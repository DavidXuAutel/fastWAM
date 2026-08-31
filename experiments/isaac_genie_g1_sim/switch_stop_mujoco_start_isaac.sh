#!/usr/bin/env bash
# On the robot: stop MuJoCo scripts under ~/mujoco_g1_infer, then start Isaac Sim (Docker headless) in background.
set -euo pipefail
MUJ="${HOME}/mujoco_g1_infer"
ISAAC_RUN="${HOME}/docker/isaac-sim/run_isaac_headless_livestream.sh"
LOG=/tmp/isaac_headless.log

echo "==> Stop MuJoCo (viewer / client from $MUJ)"
for pat in \
  "${MUJ}/launch_genie_g1_viewer.py" \
  "${MUJ}/loop_mujoco_g1_inference_client.py" \
  "${MUJ}/start_genie_g1_viewer.sh" \
  "${MUJ}/verify_mujoco_env.py" \
  ; do
  pkill -f "$pat" 2>/dev/null || true
done
sleep 1
pgrep -af "mujoco_g1_infer" || echo "(no matching mujoco_g1_infer processes)"

echo "==> Start Isaac Sim headless"
log_fail() {
  {
    date -Is
    echo "$1"
    echo "groups: $(groups 2>&1)"
    echo "--- docker info (first lines) ---"
    (timeout 8 docker info 2>&1 || docker info 2>&1) | head -20
  } >>"$LOG" 2>&1
  echo "$1" >&2
}

if [[ ! -x "$ISAAC_RUN" ]]; then
  : >"$LOG" 2>/dev/null || true
  log_fail "ERROR: missing $ISAAC_RUN — run: bash ~/isaac_genie_g1_sim/install_isaac_sim_docker_ubuntu.sh volumes"
  exit 2
fi
docker_ok() { docker info >/dev/null 2>&1; }

: >"$LOG"
if docker_ok; then
  nohup bash "$ISAAC_RUN" >>"$LOG" 2>&1 &
elif command -v sg >/dev/null 2>&1 && sg docker -c "docker info" >/dev/null 2>&1; then
  echo "Note: using sg docker (group active without full re-login)." | tee -a "$LOG" >&2
  nohup sg docker -c "bash '$ISAAC_RUN'" >>"$LOG" 2>&1 &
else
  log_fail "ERROR: docker not usable. Fix: sudo usermod -aG docker \"\$USER\" then log out and SSH again. Or: sg docker -c 'docker info' must succeed."
  exit 3
fi

echo "Isaac headless started pid=$!  log=$LOG"
sleep 2
tail -n 30 "$LOG" || true
