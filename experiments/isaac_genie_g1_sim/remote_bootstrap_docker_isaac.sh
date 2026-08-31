#!/usr/bin/env bash
# Run ON the robot as yao; optional stdin: sudo password (same line as SSH if needed).
# Adds user to docker group, fixes Isaac volume ownership, pulls Isaac image (large).
set -euo pipefail
IMAGE="${ISAAC_SIM_DOCKER_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
ROOT="${ISAAC_DOCKER_DATA_ROOT:-$HOME/docker/isaac-sim}"
SUDO_PASS="${ISAAC_DEPLOY_SUDO_PASS:-}"

sudo_s() {
  if [[ -n "${SUDO_PASS}" ]]; then
    echo "${SUDO_PASS}" | sudo -SE "$@"
  else
    sudo -n "$@" 2>/dev/null || { echo "Need sudo: set ISAAC_DEPLOY_SUDO_PASS or run sudo -v first" >&2; return 1; }
  fi
}

echo "==> docker group for ${USER}"
sudo_s usermod -aG docker "${USER}" || true

echo "==> chown 1234:1234 ${ROOT}"
sudo_s chown -R 1234:1234 "${ROOT}"

echo "==> docker pull ${IMAGE} (long)"
sudo_s docker pull "${IMAGE}"

echo "OK bootstrap pull done. Re-SSH for docker group; then: bash ~/isaac_genie_g1_sim/install_isaac_sim_docker_ubuntu.sh smoke"
