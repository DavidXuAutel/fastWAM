# G1 Camera Topic Blocker Report (FastWAM Integration)

## Goal

Enable these ROS2 topics on G1 for FastWAM:

- `/camera/head_color`
- `/camera/hand_left_color`
- `/camera/hand_right_color`

## Current Result

Not fully achievable on current robot image/config due two independent blockers:

1. Head RGB device (`d455_1`) is unavailable at hardware/driver level.
2. `forwarder` binary does not bridge hand color topics from internal bus to ROS2.

## Verified Facts

### 1) Mode switch request is accepted and RGB config is generated

- `a2d_mode_switch_server` returns `success: true` for `camera_model: "develop"` with `img_preproc` for `head`, `hand_left`, `hand_right`.
- `/home/agi/app/conf/deploy/develop/rs_camera_conf.json` is generated with:
  - `d455_1 -> head`
  - `d405_1 -> hand_left`
  - `d405_2 -> hand_right`

### 2) Internal camera pipeline publishes hand color, but not head color

From `/tmp/data/logs/cosine_runner.INFO`:

- Publishers created:
  - `/camera/hand_left_color`
  - `/camera/hand_right_color`
- Errors for head RGB:
  - `Missing SN for device: d455_1`
  - `RealSense error for device d455_1: ... Device or resource busy`

### 3) ROS2 side still misses hand color topics

`ros2 topic list` on robot does not include:

- `/camera/hand_left_color`
- `/camera/hand_right_color`
- `/camera/head_color`

Only compressed fisheye topics appear when enabled:

- `/camera/head_center_fisheye`
- `/camera/head_left_fisheye`
- `/camera/head_right_fisheye`

### 4) forwarder binary only exposes head fisheye camera bridge

`strings /home/agi/app/bin/forwarder` contains:

- `need_compressed_fisheye`
- `/camera/head_center_fisheye`
- `/camera/head_left_fisheye`
- `/camera/head_right_fisheye`

No `hand_left_color` / `hand_right_color` / `head_color` forwarding entries were found.

### 5) Hardware inventory shows no D455 attached

Even after stopping `genie_app.service`:

- `lsusb | grep 8086` shows two Intel D405 devices only.
- `rs-enumerate-devices` enumerates two D405, no D455.

This explains why `head_color` cannot be restored via software changes alone.

## Attempts Executed

- Switched scenes (`develop`, `copilot`, `preprocess71`), re-validated topic list each time.
- Patched manifest to pass `forwarder_config.yaml` in copilot scene.
- Toggled `need_compressed_fisheye` in `forwarder_config.yaml` (`false/true`).
- Restarted `genie_app.service` and reissued mode-switch requests.
- Verified process occupancy of `/dev/video*` and RealSense enumeration.

## Why This Is Blocked

- `head_color`: blocked by missing/non-functional D455 device on this unit.
- `hand_*_color`: available internally in cosine, but dropped before ROS2 because current `forwarder` build does not publish those streams to ROS topics.

## Required Upstream Fixes

1. Restore head camera hardware path (`d455_1`) or update driver/serial mapping.
2. Provide forwarder build/config that exports:
   - `/camera/hand_left_color`
   - `/camera/hand_right_color`
   - (optionally) `/camera/head_color` once D455 is restored

## Robot State After Investigation

- Scene restored to `develop`.
- `genie_app.service` active.
- No code changes were applied to FastWAM policy logic in this step.
