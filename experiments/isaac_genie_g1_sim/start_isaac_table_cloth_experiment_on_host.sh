#!/usr/bin/env bash
# On robot: build table_cloth_env.usd if missing, then start Isaac Sim and load it.
#
# Usage:
#   bash start_isaac_table_cloth_experiment_on_host.sh --headless
#   bash start_isaac_table_cloth_experiment_on_host.sh --gui
#   bash start_isaac_table_cloth_experiment_on_host.sh --skip-build --gui
#
# Env:
#   ISAAC_SIM_PATH — Isaac root with python.sh (required unless ISAAC_USE_DOCKER=1 on load).
#   ISAAC_SIM_EXPORT_ROOT — host dir for generated USD (default ~/isaac_sim_exports).
#   ISAAC_USE_DOCKER — set to 1 only if build/load via Docker (optional).
#   ISAAC_DEPLOY_SUDO_PASS — only when ISAAC_USE_DOCKER=1 and sudo docker needed.
#   ISAAC_EXPERIMENT_STEPS — headless tick count (default 400)
#   DISPLAY — required for --gui when running over SSH without physical session
set -euo pipefail
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_ROOT="${ISAAC_SIM_EXPORT_ROOT:-${HOME}/isaac_sim_exports}"
USD="${ISAAC_EXPERIMENT_USD:-${EXPORT_ROOT}/table_cloth_env.usd}"
MODE=headless
SKIP_BUILD=0
STEPS="${ISAAC_EXPERIMENT_STEPS:-400}"
EXTRA_SKIP=()

_usage() {
  echo "usage: $0 [--headless|--gui] [--skip-build] [--usd PATH] [--steps N]" >&2
  echo "  Requires native Isaac (python.sh). USD defaults to ~/isaac_sim_exports/table_cloth_env.usd." >&2
  echo "  --gui        Window + scene (uses --interactive; needs DISPLAY)." >&2
  echo "  --headless   No window (default)." >&2
  echo "  --skip-build Do not invoke run_build_* if USD is missing." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gui) MODE=gui; shift ;;
    --headless) MODE=headless; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --usd) USD="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --build-skip-cloth) EXTRA_SKIP=(skip); shift ;;
    -h|--help) _usage; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2
      _usage
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$USD")"

if [[ ! -f "$USD" ]]; then
  if [[ "$SKIP_BUILD" -eq 1 ]]; then
    echo "Missing USD and --skip-build: $USD" >&2
    exit 1
  fi
  echo "Building experiment USD → $USD"
  if [[ ${#EXTRA_SKIP[@]} -gt 0 ]]; then
    bash "${SDIR}/run_build_table_cloth_env_on_host.sh" "$USD" "${EXTRA_SKIP[@]}"
  else
    bash "${SDIR}/run_build_table_cloth_env_on_host.sh" "$USD"
  fi
fi

LOAD_ARGS=(--usd "$USD")
if [[ "$MODE" == "gui" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "WARN: DISPLAY is empty; GUI may fail. Try: export DISPLAY=:0 or use SSH -X/-Y." >&2
  fi
  LOAD_ARGS+=(--interactive)
else
  LOAD_ARGS+=(--headless --steps "${STEPS}")
fi

exec bash "${SDIR}/run_load_table_cloth_on_host.sh" "${LOAD_ARGS[@]}"
