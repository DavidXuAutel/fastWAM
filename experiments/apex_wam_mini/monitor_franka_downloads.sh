#!/usr/bin/env bash
# Monitor Franka open-dataset downloads: poll every 5 min, retry stalled targets,
# write data/franka/.ALL_DONE when all 8 datasets are present.
set -euo pipefail

ROOT="/home/a25689/FastWAM"
cd "$ROOT"
FRANKA_DIR="data/franka"
DONE_MARKER="${FRANKA_DIR}/.ALL_DONE"
DOWNLOAD_LOG="data/apex_wam_mini/reports/franka_download.log"
MONITOR_LOG="data/apex_wam_mini/reports/franka_monitor.log"
INTERVAL=300  # 5 minutes

MONITOR_LOCK="data/apex_wam_mini/reports/monitor_franka_downloads.lock"
exec 8>"$MONITOR_LOCK"
if ! flock -n 8; then
  echo "[$(date -Iseconds)] monitor_franka_downloads already running; exit" >&2
  exit 0
fi

# repo_id | download_franka target_arg | out_dir (under FRANKA_DIR)
TARGETS=(
  "libero_90|libero_90|${FRANKA_DIR}/libero_90"
  "lerobot/libero|lerobot_libero|${FRANKA_DIR}/lerobot_libero"
  "lerobot/libero_plus|libero_plus|${FRANKA_DIR}/libero_plus"
  "robbyant/libero-long-lerobot|libero_long|${FRANKA_DIR}/libero_long"
  "Topasm/Franka_move|franka_move|${FRANKA_DIR}/franka_move"
  "lerobot/cmu_franka_exploration_dataset|cmu_franka|${FRANKA_DIR}/cmu_franka_exploration"
  "IPEC-COMMUNITY/droid_lerobot|droid_lerobot|${FRANKA_DIR}/droid_lerobot"
  "lerobot/droid_1.0.1|droid_1.0.1|${FRANKA_DIR}/droid_1.0.1"
)

mlog() { echo "[$(date -Iseconds)] $*" | tee -a "$MONITOR_LOG" >&2; }

dataset_ready() {
  local out="$1"
  [[ -f "${out}/meta/info.json" ]]
}

all_done() {
  [[ -f "$DONE_MARKER" ]] && return 0
  for t in "${TARGETS[@]}"; do
    local out
    out="$(printf '%s' "$t" | cut -d'|' -f3)"
    dataset_ready "$out" || return 1
  done
  return 0
}

remaining_targets() {
  for t in "${TARGETS[@]}"; do
    local out
    out="$(printf '%s' "$t" | cut -d'|' -f3)"
    dataset_ready "$out" || printf '%s\n' "$t"
  done
}

download_proc_running() {
  pgrep -f '[d]ownload_franka_datasets.sh' >/dev/null 2>&1
}

status_summary() {
  local ready=0 total="${#TARGETS[@]}"
  for t in "${TARGETS[@]}"; do
    local out
    out="$(printf '%s' "$t" | cut -d'|' -f3)"
    dataset_ready "$out" && ready=$((ready + 1))
  done
  mlog "progress ${ready}/${total} sizes: $(du -sh ${FRANKA_DIR}/* 2>/dev/null | tr '\n' ' ')"
}

launch_download() {
  mlog "RETRY/launch download_franka_datasets.sh all"
  nohup bash experiments/apex_wam_mini/download_franka_datasets.sh all \
    >> "$DOWNLOAD_LOG" 2>&1 &
  echo $!
}

mkdir -p "$FRANKA_DIR" data/apex_wam_mini/reports
mlog "monitor_franka started (interval=${INTERVAL}s)"

while true; do
  if all_done; then
    mlog "ALL 8 FRANKA DATASETS COMPLETE"
    date -Iseconds > "$DONE_MARKER"
    break
  fi

  if download_proc_running; then
    status_summary
  else
    remaining="$(remaining_targets)"
    if [[ -z "$remaining" ]]; then
      mlog "all datasets ready and proc idle -> marking complete"
      date -Iseconds > "$DONE_MARKER"
      break
    fi
    mlog "download idle; remaining=$(printf '%s\n' "$remaining" | cut -d'|' -f1 | tr '\n' ',')"
    launch_download
  fi

  sleep "$INTERVAL"
done

mlog "monitor_franka exiting"
