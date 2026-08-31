#!/usr/bin/env bash
# Invoked on robot via ssh (no pipes on ssh argv — avoids expect/spawn quoting bugs).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/isaac_env_check_last.log"
: >"$LOG"
bash "$DIR/isaac_env_check.sh" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
