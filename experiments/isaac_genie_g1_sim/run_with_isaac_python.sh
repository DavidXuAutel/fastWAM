#!/usr/bin/env bash
# Run a Python script with Isaac Sim's interpreter (required for omni/isaacsim imports).
set -euo pipefail
ISA="${ISAAC_SIM_PATH:-}"
if [[ -z "$ISA" ]]; then
  echo "Set ISAAC_SIM_PATH to your Isaac Sim root (directory containing python.sh)." >&2
  exit 2
fi
PY="$ISA/python.sh"
if [[ ! -x "$PY" ]]; then
  echo "Not found or not executable: $PY" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$@"
