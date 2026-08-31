#!/usr/bin/env bash
# On G1 (upload to /tmp/g1_run_arm_delta_root.sh):
#   echo 1 | sudo -S bash /tmp/g1_run_arm_delta_root.sh --delta-deg 5 --side left
unset PYTHONPATH
export PATH="/opt/ros/humble/bin:/usr/bin:/bin"
set +u
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
exec python3 /tmp/g1_arm_joint_delta.py "$@"
