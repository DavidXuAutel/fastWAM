#!/usr/bin/env bash
# Reverse tunnel: H100:8010 -> yao@10.229.20.125:8010
# Lab PC talks to FR3 FCI at 10.229.66.91; do not change Desk / 10.229.66.70.
set -euo pipefail

H100_HOST="${H100_HOST:-10.239.121.11}"
H100_PORT="${H100_PORT:-31126}"
H100_USER="${H100_USER:-a25689}"
LAB_HOST="${LAB_HOST:-10.229.20.125}"
LAB_USER="${LAB_USER:-yao}"
SCOUT_PORT="${SCOUT_PORT:-8010}"
MAC_KEY="${MAC_KEY:-$HOME/.ssh/franka_ros2_ed25519}"
TUNNEL_KEY_ON_H100="${TUNNEL_KEY_ON_H100:-/home/a25689/.ssh/kairos_h100_to_125_ed25519}"

ssh_h100() {
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$MAC_KEY" -p "$H100_PORT" \
    "${H100_USER}@${H100_HOST}" "$@"
}

ssh_lab() {
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$MAC_KEY" \
    "${LAB_USER}@${LAB_HOST}" "$@"
}

cmd="${1:-status}"
case "$cmd" in
  start)
    ssh_h100 "bash -s" <<EOF
set -euo pipefail
pkill -f 'ssh .* -R ${SCOUT_PORT}:127.0.0.1:${SCOUT_PORT} ${LAB_USER}@${LAB_HOST}' 2>/dev/null || true
sleep 1
test -f ${TUNNEL_KEY_ON_H100}
ssh -fN -o ExitOnForwardFailure=yes -o BatchMode=yes -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -i ${TUNNEL_KEY_ON_H100} \
  -R ${SCOUT_PORT}:127.0.0.1:${SCOUT_PORT} ${LAB_USER}@${LAB_HOST}
pgrep -af 'ssh .* -R ${SCOUT_PORT}:127.0.0.1:${SCOUT_PORT}' | head -3
EOF
    sleep 2
    ssh_lab "curl -4 -sS --noproxy '*' -m 30 http://127.0.0.1:${SCOUT_PORT}/health"; echo
    ;;
  stop)
    ssh_h100 "pkill -f 'ssh .* -R ${SCOUT_PORT}:127.0.0.1:${SCOUT_PORT} ${LAB_USER}@${LAB_HOST}' || true"
    echo "stopped"
    ;;
  status)
    echo "=== H100 Scout ==="
    ssh_h100 "curl -4 -sS --noproxy '*' -m 15 http://127.0.0.1:${SCOUT_PORT}/health; echo; pgrep -af 'ssh .* -R ${SCOUT_PORT}:127.0.0.1:${SCOUT_PORT}' || echo 'no reverse tunnel'"
    echo "=== LAB ${LAB_HOST} ==="
    ssh_lab "curl -4 -sS --noproxy '*' -m 30 http://127.0.0.1:${SCOUT_PORT}/health; echo; ping -c 1 -W 1 10.229.66.91 | tail -2"
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
