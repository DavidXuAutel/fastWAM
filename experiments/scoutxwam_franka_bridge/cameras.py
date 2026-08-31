#!/usr/bin/env python3
"""Dual-cam helpers for ScoutXWAM (exterior then wrist, 256x320 each)."""
from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

# Match scoutxwam_droid100 training video_size [H, W]
SCOUT_H = 256
SCOUT_W = 320

CAM1_COMPRESSED = "/cam1/cam1/color/image_raw/compressed"  # wrist
CAM2_COMPRESSED = "/cam2/cam2/color/image_raw/compressed"  # scene / exterior


def decode_compressed_image(msg) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(bytes(msg.data))).convert("RGB"), dtype=np.uint8)


def realsense_image_qos():
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )

    reliable_tl = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    reliable_vol = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )
    return (reliable_tl, reliable_vol, qos_profile_sensor_data)


def resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {getattr(image, 'shape', None)}")
    pil = Image.fromarray(image).convert("RGB")
    out = pil.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(out, dtype=np.uint8)


def stack_exterior_wrist(
    exterior_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    *,
    height: int = SCOUT_H,
    width: int = SCOUT_W,
) -> np.ndarray:
    """Return uint8 video [2, H, W, 3] with view0=exterior, view1=wrist."""
    v0 = resize_rgb(exterior_rgb, height, width)
    v1 = resize_rgb(wrist_rgb, height, width)
    return np.stack([v0, v1], axis=0)
