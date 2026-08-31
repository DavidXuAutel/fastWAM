#!/usr/bin/env bash
set +e
echo "=== id / groups ==="
id
groups | tr ' ' '\n' | grep -x docker && echo "(user in docker group)" || echo "(user NOT in docker group — run: sudo usermod -aG docker \$USER && re-login)"
echo "=== docker info (15s cap) ==="
if command -v timeout >/dev/null 2>&1; then
  timeout 15 docker info 2>&1 | head -30
else
  docker info 2>&1 | head -30
fi
echo "=== docker images (isaac) ==="
timeout 20 docker images 2>&1 | grep -iE 'isaac|REPOSITORY' | head -15 || docker images 2>&1 | head -8
echo "=== runner ==="
test -f "$HOME/docker/isaac-sim/run_isaac_headless_livestream.sh" && echo "OK runner exists" || echo "MISSING runner — run install_isaac_sim_docker_ubuntu.sh volumes"
echo "=== headless log ==="
test -f /tmp/isaac_headless.log && tail -n 15 /tmp/isaac_headless.log || echo "(no /tmp/isaac_headless.log)"
