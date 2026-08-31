#!/usr/bin/env bash
# FastWAM / Genie G1：加载 ROS 2 Humble 全局环境变量（在当前 shell 中执行）
#
# 用法（在仓库根目录或任意目录）:
#   source /path/to/FastWAM/scripts/env_ros2_humble.sh
#
# 可选：叠加智元 GDK 等工作空间（需已 colcon build）
#   export GENIE_ROS_WS=/path/to/your_ws
#   source /path/to/FastWAM/scripts/env_ros2_humble.sh
#
# 常用覆盖（在 source 本脚本之前 export 即可）:
#   ROS_HUMBLE_PREFIX  默认 /opt/ros/humble
#   ROS_DISTRO         默认 humble
#   RMW_IMPLEMENTATION 默认 rmw_fastrtps_cpp
#   ROS_DOMAIN_ID      默认 0
#   ROS_HUMBLE_CONDA_ENV  macOS 回退使用的 conda env（默认 ros2_humble39）

set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
export ROS_DISTRO

ROS_HUMBLE_PREFIX="${ROS_HUMBLE_PREFIX:-/opt/ros/${ROS_DISTRO}}"
export ROS_HUMBLE_PREFIX
_using_conda_ros=0

_setup="${ROS_HUMBLE_PREFIX}/setup.bash"
if [[ -f "${_setup}" ]]; then
  # Ubuntu / Debian official layout.
  # ROS setup scripts may read unset variables while expanding chained hooks.
  set +u
  # shellcheck source=/dev/null
  source "${_setup}"
  set -u
else
  # macOS fallback: use RoboStack Humble from conda.
  _using_conda_ros=1
  ROS_HUMBLE_CONDA_ENV="${ROS_HUMBLE_CONDA_ENV:-ros2_humble39}"
  export ROS_HUMBLE_CONDA_ENV

  _conda_bin="${CONDA_EXE:-}"
  if [[ -z "${_conda_bin}" ]]; then
    if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
      _conda_bin="${HOME}/miniconda3/bin/conda"
    elif command -v conda >/dev/null 2>&1; then
      _conda_bin="$(command -v conda)"
    fi
  fi

  if [[ -z "${_conda_bin}" ]]; then
    echo "错误: 未找到 ${_setup}，且未检测到 conda。" >&2
    echo "请先安装 ROS2 Humble（Linux）或创建 RoboStack conda env（macOS）。" >&2
    return 1 2>/dev/null || exit 1
  fi

  _conda_base="$("${_conda_bin}" info --base 2>/dev/null || true)"
  if [[ -z "${_conda_base}" || ! -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
    echo "错误: conda base 无效，无法激活 ${ROS_HUMBLE_CONDA_ENV}。" >&2
    return 1 2>/dev/null || exit 1
  fi

  # conda.sh may touch unset vars; temporarily relax nounset.
  set +u
  # shellcheck source=/dev/null
  source "${_conda_base}/etc/profile.d/conda.sh"
  if ! conda activate "${ROS_HUMBLE_CONDA_ENV}" >/dev/null 2>&1; then
    set -u
    echo "错误: conda 环境 ${ROS_HUMBLE_CONDA_ENV} 不存在。" >&2
    echo "可执行: conda create -n ${ROS_HUMBLE_CONDA_ENV} python=3.9 并安装 ros-humble-ros-base。" >&2
    return 1 2>/dev/null || exit 1
  fi
  set -u
fi

if [[ "${_using_conda_ros}" -eq 1 ]]; then
  _rmw_default="rmw_cyclonedds_cpp"
else
  _rmw_default="rmw_fastrtps_cpp"
fi
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-${_rmw_default}}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

if [[ -n "${GENIE_ROS_WS:-}" ]]; then
  _genie_setup="${GENIE_ROS_WS}/install/setup.bash"
  if [[ -f "${_genie_setup}" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "${_genie_setup}"
    set -u
  else
    echo "警告: GENIE_ROS_WS=${GENIE_ROS_WS} 下未找到 install/setup.bash，已跳过叠加。" >&2
  fi
fi

echo "ROS 2 已加载: ROS_DISTRO=${ROS_DISTRO} ROS_HUMBLE_PREFIX=${ROS_HUMBLE_PREFIX} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION} ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
