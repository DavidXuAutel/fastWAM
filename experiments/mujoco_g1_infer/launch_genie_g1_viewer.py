#!/usr/bin/env python3
"""
Launch the MuJoCo **Simulate** interactive viewer with a Genie G01 MJCF.

Default scene: ``scenes/genie_g1_arm14_proxy.xml`` (14-DoF arm proxy).
Override with ``--mjcf`` or env ``MUJOCO_GENIE_G1_XML`` (vendor MJCF).

Requires a display and GLFW (typical: local desktop, or ``ssh -X/-Y``, or ``xvfb-run`` for smoke tests).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MJCF = _SCRIPT_DIR / "scenes" / "genie_g1_arm14_proxy.xml"


def main() -> int:
    ap = argparse.ArgumentParser(description="MuJoCo Simulate viewer — Genie G01 MJCF")
    ap.add_argument(
        "--mjcf",
        default=os.environ.get("MUJOCO_GENIE_G1_XML"),
        help="Main MJCF path (default: bundled proxy or MUJOCO_GENIE_G1_XML)",
    )
    args = ap.parse_args()
    path = Path(args.mjcf).expanduser().resolve() if args.mjcf else _DEFAULT_MJCF
    if not path.is_file():
        print(f"MJCF not found: {path}", file=sys.stderr)
        return 1

    if not os.environ.get("DISPLAY") and sys.platform == "linux":
        print(
            "Note: DISPLAY is unset. Use a desktop session, `ssh -X/-Y`, or e.g.\n"
            "  xvfb-run -a python3 launch_genie_g1_viewer.py\n",
            file=sys.stderr,
        )

    try:
        import mujoco.viewer
    except RuntimeError as e:
        print(f"Cannot open viewer (GLFW): {e}", file=sys.stderr)
        return 2

    print("Loading:", path)
    mujoco.viewer.launch_from_path(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
