# G1 Camera Escalation Ticket (FastWAM)

## Request

Please help restore ROS2 camera topics required by FastWAM:

- `/camera/head_color`
- `/camera/hand_left_color`
- `/camera/hand_right_color`

## Environment

- Robot: G1 (`10.229.66.60`)
- Scene switch server: `a2d_mode_switch_server` (`tcp://10.42.0.101:8848`)
- Current scene used for tests: `develop`
- Service: `genie_app.service` (active)

## Repro Steps

1. Send mode-switch request:
   - `mode: COPILOT`
   - `hybrid_deploy_config.camera_model: "develop"`
   - `img_preproc`: `head`, `hand_left`, `hand_right`
2. Confirm request response: `success: true`
3. Confirm generated config:
   - `/home/agi/app/conf/deploy/develop/rs_camera_conf.json`
   - Contains `d455_1`, `d405_1`, `d405_2`
4. Check ROS2:
   - `ros2 topic list | grep /camera`
   - Missing required color topics

## Observed Facts

### A) Internal cosine bus has hand color publishers

From `/tmp/data/logs/cosine_runner.INFO`:

- publisher created: `/camera/hand_left_color`
- publisher created: `/camera/hand_right_color`

So hand color exists internally.

### B) Head color fails in cosine

Same log shows:

- `Missing SN for device: d455_1`
- `RealSense error for device d455_1 ... Device or resource busy`

### C) USB/hardware inventory shows no D455

After stopping `genie_app.service`:

- `lsusb | grep 8086` shows only two D405 devices
- `rs-enumerate-devices` shows two D405 devices, no D455

### D) forwarder does not bridge hand color to ROS2

`strings /home/agi/app/bin/forwarder` contains only fisheye camera ROS strings:

- `/camera/head_center_fisheye`
- `/camera/head_left_fisheye`
- `/camera/head_right_fisheye`

No `hand_left_color`, `hand_right_color`, `head_color` forwarding entries found.

## Additional Blocker for Temporary Relay

Robot has `/home/agi/app/python/cosine_bus_py/cosine_bus_py.py`, but no shared library:

- missing `libcosine_bus_py.so`

Without this `.so`, Python cannot subscribe internal cosine image topics for temporary ROS relay.

## What We Need From Vendor

1. **Head RGB recovery**
   - Restore D455 hardware path / serial mapping / driver init
   - Ensure `d455_1` is enumerated and stable
2. **Forwarder capability**
   - Provide build/config to expose ROS2 topics:
     - `/camera/hand_left_color`
     - `/camera/hand_right_color`
     - `/camera/head_color`
3. **Optional temporary workaround support**
   - Provide `libcosine_bus_py.so` compatible with current image for custom relay

## Useful Files

- `experiments/genie_g1/g1_camera_blocker_report.md`
- `experiments/genie_g1/g1_vendor_support_ticket.md`
