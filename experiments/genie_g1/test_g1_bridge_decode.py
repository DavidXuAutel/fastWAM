#!/usr/bin/env python3
"""Local unit checks for ros_g1_bridge image decoders (no ROS daemon required)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

# Load ros_g1_bridge from repo without ROS imports succeeding for Image types — decoders are pure.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "genie_g1"))

import ros_g1_bridge as rgb  # noqa: E402


def _tiny_jpeg_bgr() -> bytes:
    try:
        import cv2  # type: ignore

        bgr = np.zeros((16, 24, 3), dtype=np.uint8)
        bgr[:, :] = (40, 80, 120)
        ok, buf = cv2.imencode(".jpg", bgr)
        assert ok
        return bytes(buf.tobytes())
    except Exception:
        from PIL import Image  # type: ignore

        rgb_img = np.zeros((16, 24, 3), dtype=np.uint8)
        rgb_img[:, :] = (120, 80, 40)
        bio = io.BytesIO()
        Image.fromarray(rgb_img).save(bio, format="JPEG", quality=90)
        return bio.getvalue()


def main() -> None:
    class _Img:
        height = 16
        width = 24
        encoding = "bgr8"
        data: bytes

    raw = _Img()
    raw.data = np.zeros((16, 24, 3), dtype=np.uint8).tobytes()
    out = rgb.image_msg_to_rgb(raw)
    assert out.shape == (16, 24, 3)

    class _Comp:
        format = "jpeg"
        data: list[int]

    comp = _Comp()
    comp.data = list(_tiny_jpeg_bgr())
    out2 = rgb.compressed_image_msg_to_rgb(comp)
    assert out2.shape[0] >= 8 and out2.shape[2] == 3
    print("decode_ok", out.shape, out2.shape)


if __name__ == "__main__":
    main()
