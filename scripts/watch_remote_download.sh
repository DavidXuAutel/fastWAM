#!/usr/bin/env bash
# LOCAL (macOS) watcher: poll remote every 5 min for the .ALL_DONE marker,
# then fire an osascript popup notification.
set -euo pipefail
REMOTE="a25689@10.239.121.11"
PORT=31126
INTERVAL=300
PASS="123456"
MARKER="data/official/.ALL_DONE"

# Use expect to feed password for each poll.
check_done() {
  expect <<EOF 2>/dev/null
set timeout 60
log_user 0
spawn ssh -o StrictHostKeyChecking=no -p $PORT $REMOTE "cat $MARKER 2>/dev/null && echo MARKER_FOUND || echo MARKER_MISSING"
expect -re "assword:" { send "$PASS\r" }
expect {
  "MARKER_FOUND" { exit 0 }
  "MARKER_MISSING" { exit 1 }
  eof { exit 2 }
}
EOF
}

notify_popup() {
  osascript -e 'display notification "远端 LIBERO + RoboTwin 官方全量数据已全部下载完成。" with title "下载完成 ✓" sound name "Glass"'
  osascript -e 'display dialog "远端 LIBERO + RoboTwin 官方全量数据已全部下载完成！" buttons {"好的"} default button "好的" with title "下载完成 ✓"'
}

echo "[watcher] started, polling every ${INTERVAL}s"
while true; do
  if check_done; then
    echo "[watcher] marker found, firing popup"
    notify_popup
    break
  fi
  echo "[watcher] not done yet, sleeping ${INTERVAL}s ($(date -Iseconds))"
  sleep "$INTERVAL"
done
echo "[watcher] exiting"
