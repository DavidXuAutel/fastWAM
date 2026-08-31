# ROS 2 Humble Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a reproducible Ubuntu 22.04 container with ROS 2 Humble for FastWAM development and G1 ROS experiments.

**Architecture:** Add a focused Docker environment under `docker/ros2_humble/`, with a helper script in `scripts/` so users can build, enter, and verify the container from the FastWAM root. Keep the existing `scripts/env_ros2_humble.sh` contract intact by installing ROS under `/opt/ros/humble`.

**Tech Stack:** Docker Desktop on macOS, Ubuntu 22.04, ROS 2 Humble apt packages, Docker Compose.

---

### Task 1: Container Definition

**Files:**
- Create: `docker/ros2_humble/Dockerfile`
- Create: `docker/ros2_humble/compose.yml`

- [ ] Define an Ubuntu 22.04 image that installs ROS 2 Humble from the official ROS apt repository.
- [ ] Include common ROS packages needed by existing FastWAM scripts: `ros-humble-ros-base`, `ros-humble-rclpy`, `ros-humble-sensor-msgs`, `ros-humble-rosbag2`, and Fast DDS support.
- [ ] Configure `/fastwam` as the mounted project workspace.

### Task 2: Developer Entrypoint

**Files:**
- Create: `scripts/docker_ros2_humble.sh`

- [ ] Add `build`, `shell`, `run`, `verify`, `up`, and `down` subcommands.
- [ ] Keep Docker Compose invocation rooted at the FastWAM repository.
- [ ] Source `/opt/ros/humble/setup.bash` automatically for interactive shells and verification.

### Task 3: Documentation

**Files:**
- Create: `docker/ros2_humble/README.md`

- [ ] Document Docker Desktop prerequisite on macOS.
- [ ] Document build, shell, verify, and cleanup commands.
- [ ] Explain mounted workspace and ROS environment variables.

### Task 4: Verification

**Files:**
- Validate: `scripts/docker_ros2_humble.sh`
- Validate: `docker/ros2_humble/compose.yml`
- Validate: `docker/ros2_humble/Dockerfile`

- [ ] Run `bash -n scripts/docker_ros2_humble.sh`.
- [ ] Run `docker compose -f docker/ros2_humble/compose.yml config` if Docker is available.
- [ ] Run `scripts/docker_ros2_humble.sh build` if Docker Desktop is installed and running.
- [ ] Run `scripts/docker_ros2_humble.sh verify` and confirm ROS 2 Humble commands work inside the container.
