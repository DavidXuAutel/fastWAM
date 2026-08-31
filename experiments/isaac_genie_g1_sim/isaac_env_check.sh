#!/usr/bin/env bash
# Fast Isaac-on-host sanity check (run on robot).
set +e
echo "========== 1) user / docker group =========="
id
groups | tr ' ' '\n' | grep -qx docker && echo "OK: user in group 'docker'" || echo "FAIL: user NOT in group 'docker' (sudo usermod -aG docker \$USER && re-login)"

echo "========== 2) nvidia-smi =========="
command -v nvidia-smi >/dev/null && nvidia-smi -L 2>&1 | head -6 || echo "no nvidia-smi"

echo "========== 3) docker client =========="
timeout 5 docker version 2>&1 | head -12

echo "========== 4) docker info (10s cap) =========="
timeout 10 docker info 2>&1 | head -22

echo "========== 5) GPU inside container (15s) =========="
timeout 15 docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L 2>&1 | head -8 || echo "FAIL: docker --gpus (install nvidia-container-toolkit + restart docker)"

echo "========== 6) Isaac image =========="
docker images 2>&1 | grep -iE '^REPOSITORY|isaac' | head -12

echo "========== 7) helper scripts =========="
for f in "$HOME/docker/isaac-sim/run_isaac_bash_interactive.sh" "$HOME/docker/isaac-sim/run_isaac_headless_livestream.sh"; do
  if [[ -f "$f" ]]; then echo "OK $f"; head -3 "$f"; else echo "MISSING $f"; fi
done

echo "========== 8) last Isaac headless log =========="
if [[ -f /tmp/isaac_headless.log ]]; then tail -n 35 /tmp/isaac_headless.log; else echo "(no /tmp/isaac_headless.log)"; fi

echo "========== 9) docker ps (isaac) =========="
docker ps -a 2>&1 | grep -i isaac | head -8 || docker ps 2>&1 | head -6

echo "========== done =========="
