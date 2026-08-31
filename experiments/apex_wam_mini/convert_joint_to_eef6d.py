#!/usr/bin/env python3
"""Stub: convert joint-absolute LeRobot episodes to eef6d_relative for G1 proxy data.

Full FK + rot6d pipeline depends on robot URDF and Action Interface Tests (§9).
Run after mujoco/robotwin raw export is available.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        "Not implemented yet. Export LeRobot with eef6d actions directly, "
        f"or implement FK from {args.input_root} -> {args.output_root}."
    )


if __name__ == "__main__":
    main()
