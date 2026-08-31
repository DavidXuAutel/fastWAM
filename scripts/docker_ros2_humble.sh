#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/ros2_humble/compose.yml"
SERVICE_NAME="ros2-humble"
DOCKER_BIN="${DOCKER_BIN:-docker}"
COMPOSE_STANDALONE="${COMPOSE_STANDALONE:-}"

if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
  DOCKER_DESKTOP_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
  if [[ -x "${DOCKER_DESKTOP_BIN}" ]]; then
    DOCKER_BIN="${DOCKER_DESKTOP_BIN}"
    COMPOSE_STANDALONE="/Applications/Docker.app/Contents/Resources/cli-plugins/docker-compose"
    export PATH="$(dirname "${DOCKER_DESKTOP_BIN}"):${PATH}"
    export DOCKER_CLI_PLUGIN_EXTRA_DIRS="/Applications/Docker.app/Contents/Resources/cli-plugins${DOCKER_CLI_PLUGIN_EXTRA_DIRS:+:${DOCKER_CLI_PLUGIN_EXTRA_DIRS}}"
    if [[ -z "${DOCKER_HOST:-}" && -S "${HOME}/.docker/run/docker.sock" ]]; then
      export DOCKER_HOST="unix://${HOME}/.docker/run/docker.sock"
    fi
    if [[ -z "${DOCKER_CONFIG:-}" ]]; then
      export DOCKER_CONFIG="${TMPDIR:-/tmp}/fastwam-docker-config"
      mkdir -p "${DOCKER_CONFIG}"
      [[ -f "${DOCKER_CONFIG}/config.json" ]] || printf '{}\n' > "${DOCKER_CONFIG}/config.json"
    fi
  fi
fi

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

usage() {
  cat <<'EOF'
Usage: scripts/docker_ros2_humble.sh <command> [args...]

Commands:
  build          Build the Ubuntu 22.04 + ROS 2 Humble image.
  shell          Open an interactive shell with ROS 2 Humble sourced.
  run <command>  Run a command in the ROS 2 Humble container.
  verify         Verify ROS 2 Humble imports and CLI commands.
  config         Render the Docker Compose configuration.
  up             Start the container in the background.
  down           Stop and remove the container.

Environment overrides:
  ROS_DOMAIN_ID=0
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp
EOF
}

require_docker() {
  if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
    echo "error: docker command not found. Install and start Docker Desktop first." >&2
    return 1
  fi

  if ! "${DOCKER_BIN}" compose version >/dev/null 2>&1 && [[ ! -x "${COMPOSE_STANDALONE}" ]]; then
    echo "error: Docker Compose v2 is unavailable. Update Docker Desktop." >&2
    return 1
  fi
}

compose() {
  if "${DOCKER_BIN}" compose version >/dev/null 2>&1; then
    "${DOCKER_BIN}" compose -f "${COMPOSE_FILE}" --project-directory "${REPO_ROOT}" "$@"
  else
    "${COMPOSE_STANDALONE}" -f "${COMPOSE_FILE}" --project-directory "${REPO_ROOT}" "$@"
  fi
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 2
fi
shift || true

case "${command}" in
  build)
    require_docker
    compose build "${SERVICE_NAME}"
    ;;
  shell)
    require_docker
    compose run --rm "${SERVICE_NAME}" bash -lc \
      'source /fastwam/scripts/env_ros2_humble.sh && exec bash'
    ;;
  run)
    require_docker
    if [[ "$#" -eq 0 ]]; then
      echo "error: run requires a command." >&2
      usage
      exit 2
    fi
    compose run --rm "${SERVICE_NAME}" bash -lc \
      'source /fastwam/scripts/env_ros2_humble.sh && exec "$@"' bash "$@"
    ;;
  verify)
    require_docker
    compose run --rm "${SERVICE_NAME}" bash -lc \
      'set -euo pipefail
       source /fastwam/scripts/env_ros2_humble.sh
       ros2 --help >/dev/null
       python3 -c "import rclpy; from sensor_msgs.msg import JointState; print(\"rclpy and sensor_msgs import OK\")"
       ros2 topic list >/tmp/ros2_topics.txt
       echo "ROS 2 Humble verification OK"'
    ;;
  config)
    require_docker
    compose config
    ;;
  up)
    require_docker
    compose up -d "${SERVICE_NAME}"
    ;;
  down)
    require_docker
    compose down
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown command: ${command}" >&2
    usage
    exit 2
    ;;
esac
