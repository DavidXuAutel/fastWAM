#!/usr/bin/env bash
# Wrapper for root execution on G1 (upload to /tmp/g1_run_wheel_root.sh).
unset PYTHONPATH
export PATH="/opt/ros/humble/bin:/usr/bin:/bin"
set +u
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
exec python3 /tmp/g1_wheel_step_back.py "$@"
