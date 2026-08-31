#!/usr/bin/env bash
# Run on the robot (after scripts live in ~/isaac_genie_g1_sim/).
# Default: native Isaac Sim only — writes USD under ISAAC_SIM_EXPORT_ROOT (host disk, not Docker volumes).
# Optional Docker build: export ISAAC_USE_DOCKER=1 (see install_isaac_sim_docker_ubuntu.sh).
set -euo pipefail
SCRIPT="${HOME}/isaac_genie_g1_sim/build_table_cloth_env_standalone.py"
EXPORT_ROOT="${ISAAC_SIM_EXPORT_ROOT:-${HOME}/isaac_sim_exports}"
HOST_OUT="${1:-${EXPORT_ROOT}/table_cloth_env.usd}"
SKIP_FLAG="${2:-}" # optional: pass word "skip" to add --skip-cloth-physics

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing $SCRIPT — sync isaac_genie_g1_sim to this host first." >&2
  exit 1
fi

mkdir -p "$(dirname "$HOST_OUT")"
EXTRA=()
if [[ "$SKIP_FLAG" == "skip" ]]; then
  EXTRA+=(--skip-cloth-physics)
fi

_resolve_isaac_root() {
  if [[ -n "${ISAAC_SIM_PATH:-}" && -x "${ISAAC_SIM_PATH}/python.sh" ]]; then
    echo "${ISAAC_SIM_PATH}"
    return 0
  fi
  local d
  for d in \
    "${HOME}/isaac-sim" \
    "${HOME}/IsaacSim" \
    "/opt/nvidia/isaac-sim" \
    "/isaac-sim" \
    "/usr/local/isaac-sim"; do
    if [[ -x "${d}/python.sh" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

_docker_bin() {
  if docker info &>/dev/null; then
    echo docker
    return 0
  fi
  if [[ -n "${ISAAC_DEPLOY_SUDO_PASS:-}" ]] && command -v sudo &>/dev/null; then
    if echo "${ISAAC_DEPLOY_SUDO_PASS}" | sudo -SE docker info &>/dev/null; then
      echo "sudo_n docker"
      return 0
    fi
  fi
  if sudo -n docker info &>/dev/null; then
    echo "sudo_n docker"
    return 0
  fi
  return 1
}

if ISA_ROOT="$(_resolve_isaac_root)"; then
  echo "Using native Isaac: ${ISA_ROOT}/python.sh"
  exec "${ISA_ROOT}/python.sh" "$SCRIPT" --headless --out "$HOST_OUT" "${EXTRA[@]:-}"
fi

if [[ "${ISAAC_USE_DOCKER:-0}" != "1" ]]; then
  echo "ERROR: Native Isaac Sim not found (no usable python.sh under ISAAC_SIM_PATH or common paths)." >&2
  echo "  Install Isaac on this machine: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html" >&2
  echo "  Then: export ISAAC_SIM_PATH=/path/to/isaac-sim   # directory containing python.sh" >&2
  echo "  USD output defaults to: ${EXPORT_ROOT}/table_cloth_env.usd (override with arg1 or ISAAC_SIM_EXPORT_ROOT)." >&2
  echo "  If you intentionally use Docker: export ISAAC_USE_DOCKER=1" >&2
  exit 1
fi

IMAGE="${ISAAC_SIM_DOCKER_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
ROOT="${ISAAC_DOCKER_DATA_ROOT:-${HOME}/docker/isaac-sim}"
IN_DOCKER_OUT="/isaac-sim/.local/share/ov/data/genie_sim/table_cloth_env.usd"
mkdir -p "${ROOT}/data/genie_sim"

if ! DB="$(_docker_bin)"; then
  echo "ERROR: ISAAC_USE_DOCKER=1 but Docker is not usable (permission / daemon)." >&2
  echo "  Fix docker access or install native Isaac and omit ISAAC_USE_DOCKER." >&2
  exit 1
fi

_run_docker() {
  if [[ "$DB" == "sudo_n docker" ]]; then
    if [[ -n "${ISAAC_DEPLOY_SUDO_PASS:-}" ]]; then
      echo "${ISAAC_DEPLOY_SUDO_PASS}" | sudo -SE docker "$@"
    else
      sudo -n docker "$@"
    fi
  else
    docker "$@"
  fi
}

echo "Using Docker image: $IMAGE ($DB)"
_run_docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "${ROOT}/cache/main:/isaac-sim/.cache:rw" \
  -v "${ROOT}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "${ROOT}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "${ROOT}/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "${ROOT}/data:/isaac-sim/.local/share/ov/data:rw" \
  -v "${ROOT}/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "${HOME}/isaac_genie_g1_sim:/workspace:ro" \
  -w /isaac-sim \
  -u 1234:1234 \
  "$IMAGE" \
  ./python.sh /workspace/build_table_cloth_env_standalone.py --headless --out "$IN_DOCKER_OUT" "${EXTRA[@]:-}"

echo "USD on host (bind-mount): ${ROOT}/data/genie_sim/table_cloth_env.usd"
if [[ "$HOST_OUT" != "${ROOT}/data/genie_sim/table_cloth_env.usd" ]]; then
  cp -f "${ROOT}/data/genie_sim/table_cloth_env.usd" "$HOST_OUT" || true
  echo "Also copied to: $HOST_OUT"
fi
