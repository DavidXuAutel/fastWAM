#!/usr/bin/env bash
# Prepare NVIDIA Isaac Sim Docker on Ubuntu (x86_64). See NVIDIA docs:
#   https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
#
# Prefer workstation install on the robot when possible; FastWAM helper scripts default to native
# ISAAC_SIM_PATH and write USD under ~/isaac_sim_exports/. Use Docker only when ISAAC_USE_DOCKER=1.
#
# Phases (default: all except pull can be skipped):
#   deps     — check nvidia-smi, docker, nvidia-container-toolkit
#   volumes  — mkdir cache dirs + chown 1234:1234 (container user; needs sudo)
#   pull     — docker pull (large; run in tmux/screen)
#   smoke    — compatibility check inside container (needs pull first)
#   all      — deps + volumes + pull + smoke
#
# Env:
#   ISAAC_SIM_DOCKER_IMAGE   default nvcr.io/nvidia/isaac-sim:5.1.0
#   ISAAC_DOCKER_DATA_ROOT   default $HOME/docker/isaac-sim
set -euo pipefail

IMAGE="${ISAAC_SIM_DOCKER_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
ROOT="${ISAAC_DOCKER_DATA_ROOT:-$HOME/docker/isaac-sim}"
PHASE="${1:-all}"

need_sudo_chown() {
  if sudo -n true 2>/dev/null; then
    sudo chown -R 1234:1234 "$ROOT"
  else
    echo "Run: sudo chown -R 1234:1234 $ROOT" >&2
    echo "(Isaac container runs as UID 1234 per NVIDIA docs.)" >&2
  fi
}

phase_deps() {
  echo "==> GPU"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA driver first." >&2
    return 1
  fi
  nvidia-smi || true
  echo "==> Docker"
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not installed. Example:" >&2
    echo "  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && sudo sh /tmp/get-docker.sh" >&2
    echo "  sudo usermod -aG docker \"\$USER\" && newgrp docker" >&2
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: cannot use Docker (permission on /var/run/docker.sock?). Fix:" >&2
    echo "  sudo usermod -aG docker \"\$USER\" && newgrp docker   # then re-SSH" >&2
    return 1
  fi
  echo "==> NVIDIA Container Toolkit (docker GPU)"
  if docker run --rm --gpus all ubuntu:22.04 nvidia-smi >/dev/null 2>&1; then
    :
  elif docker run --rm --runtime=nvidia --gpus all ubuntu:22.04 nvidia-smi >/dev/null 2>&1; then
    :
  else
    echo "WARN: GPU inside docker failed. Install NVIDIA Container Toolkit + restart docker:" >&2
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
    return 1
  fi
  echo "OK deps."
}

phase_volumes() {
  echo "==> Volume dirs under $ROOT"
  mkdir -p "$ROOT/cache/main/ov" "$ROOT/cache/main/warp" "$ROOT/cache/computecache" \
    "$ROOT/config" "$ROOT/data/documents" "$ROOT/data/Kit" "$ROOT/logs" "$ROOT/pkg"
  need_sudo_chown
  echo "OK volumes."
}

phase_pull() {
  echo "==> docker pull $IMAGE (this is large; first run can take 30–90+ minutes)"
  docker pull "$IMAGE"
  echo "OK pull."
}

phase_smoke() {
  echo "==> Compatibility check (short)"
  docker run --entrypoint bash --gpus all --rm --network=host \
    -e "ACCEPT_EULA=Y" \
    "$IMAGE" -lc './isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window' || {
    echo "WARN: compatibility script failed or image entry differs; check Isaac Sim release notes." >&2
    return 0
  }
}

write_runner_scripts() {
  mkdir -p "$ROOT"
  cat >"$ROOT/run_isaac_bash_interactive.sh" <<EOF
#!/usr/bin/env bash
# Interactive shell inside Isaac container (rootless uid 1234).
exec docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \\
  -e "PRIVACY_CONSENT=Y" \\
  -v $ROOT/cache/main:/isaac-sim/.cache:rw \\
  -v $ROOT/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \\
  -v $ROOT/logs:/isaac-sim/.nvidia-omniverse/logs:rw \\
  -v $ROOT/config:/isaac-sim/.nvidia-omniverse/config:rw \\
  -v $ROOT/data:/isaac-sim/.local/share/ov/data:rw \\
  -v $ROOT/pkg:/isaac-sim/.local/share/ov/pkg:rw \\
  -u 1234:1234 \\
  $IMAGE
EOF
  chmod +x "$ROOT/run_isaac_bash_interactive.sh"

  cat >"$ROOT/run_isaac_headless_livestream.sh" <<EOF
#!/usr/bin/env bash
# Headless Isaac with livestream (see NVIDIA docs for WebRTC client).
exec docker run --name isaac-sim-headless --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \\
  -e "PRIVACY_CONSENT=Y" \\
  -v $ROOT/cache/main:/isaac-sim/.cache:rw \\
  -v $ROOT/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \\
  -v $ROOT/logs:/isaac-sim/.nvidia-omniverse/logs:rw \\
  -v $ROOT/config:/isaac-sim/.nvidia-omniverse/config:rw \\
  -v $ROOT/data:/isaac-sim/.local/share/ov/data:rw \\
  -v $ROOT/pkg:/isaac-sim/.local/share/ov/pkg:rw \\
  -u 1234:1234 \\
  $IMAGE ./runheadless.sh -v
EOF
  chmod +x "$ROOT/run_isaac_headless_livestream.sh"
  echo "Wrote $ROOT/run_isaac_bash_interactive.sh"
  echo "Wrote $ROOT/run_isaac_headless_livestream.sh"
}

case "$PHASE" in
  deps) phase_deps ;;
  volumes) phase_volumes; write_runner_scripts ;;
  pull) phase_pull ;;
  smoke) phase_smoke ;;
  all)
    phase_deps
    phase_volumes
    write_runner_scripts
    phase_pull
    phase_smoke || true
    ;;
  runners) write_runner_scripts ;;
  *) echo "usage: $0 [deps|volumes|pull|smoke|all|runners]"; exit 2 ;;
esac
