#!/usr/bin/env bash
# On robot: native Isaac Sim only by default (USD under ~/isaac_sim_exports by default).
# Docker loading only when ISAAC_USE_DOCKER=1 (same image as optional Docker build).
# Extra args go to load_table_cloth_env_standalone.py (e.g. --headless --steps 400 --interactive).
set -euo pipefail
LOAD="${HOME}/isaac_genie_g1_sim/load_table_cloth_env_standalone.py"

EXPORT_ROOT="${ISAAC_SIM_EXPORT_ROOT:-${HOME}/isaac_sim_exports}"
ROOT="${ISAAC_DOCKER_DATA_ROOT:-${HOME}/docker/isaac-sim}"
IMAGE="${ISAAC_SIM_DOCKER_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
DEF_HOST_USD="${EXPORT_ROOT}/table_cloth_env.usd"

_resolve_isaac_root() {
  if [[ -n "${ISAAC_SIM_PATH:-}" && -x "${ISAAC_SIM_PATH}/python.sh" ]]; then
    echo "${ISAAC_SIM_PATH}"
    return 0
  fi
  local d
  for d in "${HOME}/isaac-sim" "${HOME}/IsaacSim" "/opt/nvidia/isaac-sim" "/isaac-sim" "/usr/local/isaac-sim"; do
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

_run_docker() {
  if [[ "${DB:-}" == "sudo_n docker" ]]; then
    if [[ -n "${ISAAC_DEPLOY_SUDO_PASS:-}" ]]; then
      echo "${ISAAC_DEPLOY_SUDO_PASS}" | sudo -SE docker "$@"
    else
      sudo -n docker "$@"
    fi
  else
    docker "$@"
  fi
}

_host_usd_from_args() {
  local args=("$@")
  local i=0 host_usd=""
  while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
      --usd)
        if [[ $((i + 1)) -lt ${#args[@]} ]]; then
          host_usd="${args[$((i + 1))]}"
        fi
        break
        ;;
      --usd=*)
        host_usd="${args[$i]#*=}"
        break
        ;;
    esac
    i=$((i + 1))
  done
  if [[ -z "$host_usd" ]]; then
    host_usd="$DEF_HOST_USD"
  fi
  if [[ -f "$host_usd" ]] && command -v realpath &>/dev/null; then
    realpath "$host_usd"
  elif [[ -f "$host_usd" ]]; then
    readlink -f "$host_usd" 2>/dev/null || echo "$host_usd"
  else
    echo "$host_usd"
  fi
}

_docker_rewrite_args() {
  local dock_usd="$1"
  shift
  local args=("$@")
  local out=() i
  i=0
  while [[ $i -lt ${#args[@]} ]]; do
    case "${args[$i]}" in
      --usd)
        out+=(--usd "$dock_usd")
        i=$((i + 2))
        ;;
      --usd=*)
        out+=(--usd "$dock_usd")
        i=$((i + 1))
        ;;
      *)
        out+=("${args[$i]}")
        i=$((i + 1))
        ;;
    esac
  done
  printf '%s\0' "${out[@]}"
}

if [[ ! -f "$LOAD" ]]; then
  echo "Missing $LOAD" >&2
  exit 1
fi

if ISA_ROOT="$(_resolve_isaac_root)"; then
  echo "Load via native Isaac: ${ISA_ROOT}/python.sh"
  exec "${ISA_ROOT}/python.sh" "$LOAD" "$@"
fi

if [[ "${ISAAC_USE_DOCKER:-0}" != "1" ]]; then
  echo "ERROR: Native Isaac Sim not found. Install on host and set ISAAC_SIM_PATH, or export ISAAC_USE_DOCKER=1 for container fallback." >&2
  exit 1
fi

echo "ISAAC_USE_DOCKER=1: loading via Isaac Docker (${IMAGE})."
HOST_USD="$(_host_usd_from_args "$@")"

if [[ ! -f "$HOST_USD" ]]; then
  echo "USD missing: ${HOST_USD}" >&2
  exit 1
fi

PREFIX="${ROOT}/data/"
if [[ "$HOST_USD" != "$PREFIX"* ]]; then
  echo "Docker load requires USD under ${PREFIX} (bind-mount). Got: ${HOST_USD}" >&2
  echo "  Build with Docker first, or copy USD into that tree, or use native Isaac with USD under ${EXPORT_ROOT}/." >&2
  exit 1
fi
DOCKER_USD="/isaac-sim/.local/share/ov/data${HOST_USD#"${PREFIX}"}"

if ! DB="$(_docker_bin)"; then
  echo "ISAAC_USE_DOCKER=1 but docker unavailable." >&2
  exit 1
fi

mapfile -d '' INNER_ARGS < <(_docker_rewrite_args "$DOCKER_USD" "$@")

echo "Load via Docker (${DB}): --usd → ${DOCKER_USD}"

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
  ./python.sh /workspace/load_table_cloth_env_standalone.py "${INNER_ARGS[@]}"
