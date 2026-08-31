# FastWAM ROS 2 Humble Container

This container provides Ubuntu 22.04 with ROS 2 Humble installed under `/opt/ros/humble`.
It is intended for FastWAM development and the `experiments/genie_g1` ROS scripts that expect:

```bash
source scripts/env_ros2_humble.sh
```

## macOS Prerequisite

Install and start Docker Desktop before using this environment. The `docker` command must be available in your shell:

```bash
docker --version
docker compose version
```

## Build

From the FastWAM repository root:

```bash
scripts/docker_ros2_humble.sh build
```

## Open a ROS Shell

```bash
scripts/docker_ros2_humble.sh shell
```

The repository is mounted at `/fastwam`, and the shell sources `/fastwam/scripts/env_ros2_humble.sh`.

## Run a Command

```bash
scripts/docker_ros2_humble.sh run ros2 topic list
scripts/docker_ros2_humble.sh run python3 -c "import rclpy; print(rclpy.__name__)"
```

## Verify

```bash
scripts/docker_ros2_humble.sh verify
```

This checks the ROS 2 CLI plus Python imports for `rclpy` and `sensor_msgs`.

## ROS Environment Overrides

Pass ROS settings through environment variables:

```bash
ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp scripts/docker_ros2_humble.sh shell
```

The compose file also forwards `FASTRTPS_DEFAULT_PROFILES_FILE` and `CYCLONEDDS_URI` if they are set.

## Notes

Docker Desktop on macOS runs containers inside a Linux VM. Basic ROS 2 development works normally, but DDS discovery to physical robots on the LAN can need extra network configuration. For direct robot communication, validate discovery with the target robot and consider running the same container on an Ubuntu host if multicast or host networking becomes a blocker.

## Cleanup

```bash
scripts/docker_ros2_humble.sh down
```
