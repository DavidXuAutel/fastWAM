#!/usr/bin/env python3
"""Load proxy MJCF (no Renderer) to verify pip mujoco + GL backend."""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

root = Path(__file__).resolve().parent
xml = root / "scenes" / "genie_g1_arm14_proxy.xml"
m = mujoco.MjModel.from_xml_path(str(xml))
print("OK", "nq=", m.nq, "njnt=", m.njnt, "xml=", xml)
