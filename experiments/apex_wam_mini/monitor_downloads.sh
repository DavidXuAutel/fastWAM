#!/usr/bin/env bash
# Monitor official-full downloads: poll every 5 min, retry failed/stalled
# targets, and write a completion marker when all 5 repos are done.
#
# RoboTwin2.0 raw uses aria2c (aria2_download_robotwin.sh); other targets use
# download_official_full.sh + huggingface_hub snapshot_download.
set -euo pipefail

ROOT="/home/a25689/FastWAM"
cd "$ROOT"
PYTHON="/opt/conda/bin/python3"
LOG="data/apex_wam_mini/reports/official_full_download.log"
OFFICIAL="data/official"
DONE_MARKER="${OFFICIAL}/.ALL_DONE"
MONITOR_LOG="data/apex_wam_mini/reports/official_monitor.log"
ARIA2_LIST="data/apex_wam_mini/reports/robotwin_aria2_list.txt"
ARIA2_LIST_FALLBACK="/tmp/robotwin_aria2_list.txt"
ROBOTWIN_OUT="${OFFICIAL}/robotwin2.0_raw"
INTERVAL=300  # 5 minutes

MONITOR_LOCK="data/apex_wam_mini/reports/monitor_downloads.lock"
exec 8>"$MONITOR_LOCK"
if ! flock -n 8; then
  echo "[$(date -Iseconds)] monitor_downloads already running (lock held); exit" >&2
  exit 0
fi

# repo_id | target_arg | out_dir
TARGETS=(
  "HuggingFaceVLA/libero|libero_hfvla|${OFFICIAL}/libero_hfvla_v2.1"
  "nvidia/LIBERO_LeRobot_v3|libero_nvidia|${OFFICIAL}/libero_nvidia_lerobot_v3"
  "physical-intelligence/libero|libero_raw|${OFFICIAL}/libero_raw"
  "lerobot/robotwin_unified|robotwin_lerobot|${OFFICIAL}/robotwin_unified_lerobot"
  "TianxingChen/RoboTwin2.0|robotwin_raw|${ROBOTWIN_OUT}"
)

mlog() { echo "[$(date -Iseconds)] $*" | tee -a "$MONITOR_LOG" >&2; }

repo_done_in_log() {
  local repo="$1"
  grep -q "done ${repo} ->" "$LOG" 2>/dev/null
}

aria2_list_path() {
  if [[ -f "$ARIA2_LIST" ]]; then
    printf '%s\n' "$ARIA2_LIST"
  elif [[ -f "$ARIA2_LIST_FALLBACK" ]]; then
    printf '%s\n' "$ARIA2_LIST_FALLBACK"
  else
    return 1
  fi
}

robotwin_aria2_missing_count() {
  local list missing=0 relpath
  list="$(aria2_list_path)" || { echo "no_list"; return; }
  while IFS= read -r relpath; do
    [[ -n "$relpath" ]] || continue
    if [[ ! -s "${ROBOTWIN_OUT}/${relpath}" ]]; then
      missing=$((missing + 1))
    fi
  done < <(grep '^  out=' "$list" | sed 's/^  out=//')
  echo "$missing"
}

mark_robotwin_done_in_log() {
  if repo_done_in_log "TianxingChen/RoboTwin2.0"; then
    return 0
  fi
  echo "done TianxingChen/RoboTwin2.0 -> ${ROBOTWIN_OUT} (aria2)" >> "$LOG"
}

robotwin_raw_done() {
  if repo_done_in_log "TianxingChen/RoboTwin2.0"; then
    return 0
  fi
  local missing
  missing="$(robotwin_aria2_missing_count)"
  [[ "$missing" == "0" ]]
}

target_done() {
  local t="$1" repo target_arg
  repo="${t%%|*}"
  target_arg="$(printf '%s' "$t" | cut -d'|' -f2)"
  if [[ "$target_arg" == "robotwin_raw" ]]; then
    robotwin_raw_done
  else
    repo_done_in_log "$repo"
  fi
}

hf_download_running() {
  pgrep -f "download_official_full.sh" >/dev/null 2>&1
}

robotwin_hf_running() {
  pgrep -f "TianxingChen/RoboTwin2.0.*robotwin2.0_raw" >/dev/null 2>&1 || \
    pgrep -f "snapshot_download.*RoboTwin2.0" >/dev/null 2>&1
}

robotwin_aria2_running() {
  pgrep -x aria2c >/dev/null 2>&1 && [[ -f "${ROBOTWIN_OUT}/.aria2" || -f "$(aria2_list_path 2>/dev/null || echo __none__)" ]]
}

any_download_running() {
  hf_download_running || robotwin_aria2_running || robotwin_hf_running
}

all_done() {
  [[ -f "$DONE_MARKER" ]] && return 0
  for t in "${TARGETS[@]}"; do
    target_done "$t" || return 1
  done
  return 0
}

remaining_targets() {
  for t in "${TARGETS[@]}"; do
    target_done "$t" || printf '%s\n' "$t"
  done
}

status_summary() {
  local sizes robotwin_missing=""
  sizes="$(du -sh ${OFFICIAL}/* 2>/dev/null | tr '\n' ' ')"
  if ! target_done "TianxingChen/RoboTwin2.0|robotwin_raw|${ROBOTWIN_OUT}"; then
    robotwin_missing="$(robotwin_aria2_missing_count)"
    if [[ "$robotwin_missing" != "no_list" ]]; then
      robotwin_missing=" robotwin_missing=${robotwin_missing}"
    else
      robotwin_missing=" robotwin_missing=unknown(no_list)"
    fi
  fi
  mlog "sizes: ${sizes}${robotwin_missing} aria2=$(pgrep -xc aria2c 2>/dev/null || echo 0)"
}

launch_target() {
  local t="$1" target_arg
  target_arg="$(printf '%s' "$t" | cut -d'|' -f2)"
  if [[ "$target_arg" == "robotwin_raw" ]]; then
    mlog "RETRY/launch target=robotwin_raw via aria2"
    nohup bash experiments/apex_wam_mini/aria2_download_robotwin.sh \
      >> "data/apex_wam_mini/reports/aria2_robotwin_launch.log" 2>&1 &
  else
    mlog "RETRY/launch target=${target_arg}"
    nohup bash experiments/apex_wam_mini/download_official_full.sh "$target_arg" \
      >> "$LOG" 2>&1 &
  fi
  echo $!
}

mkdir -p "$OFFICIAL" data/apex_wam_mini/reports
mlog "monitor started (interval=${INTERVAL}s, aria2-aware)"

while true; do
  if all_done; then
    mark_robotwin_done_in_log
    mlog "ALL 5 REPOS COMPLETE"
    date -Iseconds > "$DONE_MARKER"
    break
  fi

  if any_download_running; then
    status_summary
  else
    remaining="$(remaining_targets)"
    if [[ -z "$remaining" ]]; then
      mark_robotwin_done_in_log
      mlog "no remaining targets and proc idle -> marking complete"
      date -Iseconds > "$DONE_MARKER"
      break
    fi
    next="$(printf '%s\n' "$remaining" | head -1)"
    mlog "no download proc running; remaining=$(printf '%s\n' "$remaining" | tr '\n' ',')"
    launch_target "$next"
  fi

  sleep "$INTERVAL"
done

mlog "monitor exiting"
