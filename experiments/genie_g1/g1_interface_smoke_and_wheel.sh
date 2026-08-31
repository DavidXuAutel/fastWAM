#!/usr/bin/env bash
# On G1 — HAL smoke test only (ros2 bag must run as root to receive SHM DDS from motion-control):
#   echo 1 | sudo -S bash /tmp/g1_interface_smoke_and_wheel.sh
#
# For a short backward velocity burst on /mbc/wheel_command, use g1_wheel_step_back.py + g1_run_wheel_root.sh
# (see _expect_wheel_remote.exp from dev machine).
unset PYTHONPATH
export PATH="/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
set +u
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

echo "=== HAL bag sample (3s) /hal/arm_joint_state (root) ==="
rm -rf /tmp/_g1_smoke_bag
timeout 3 ros2 bag record /hal/arm_joint_state -o /tmp/_g1_smoke_bag >/dev/null 2>&1 || true
ros2 bag info /tmp/_g1_smoke_bag 2>&1 | head -22
echo "=== done ==="
