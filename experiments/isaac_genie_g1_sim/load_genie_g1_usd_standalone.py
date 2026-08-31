#!/usr/bin/env python3
"""
Minimal Isaac Sim standalone: create a stage and reference a Genie G1 USD (e.g. from GenieSimAssets).

Must be run with Isaac's python.sh — use ./run_with_isaac_python.sh load_genie_g1_usd_standalone.py ...

All omni/pxr imports happen AFTER SimulationApp starts (Isaac requirement).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _default_usd() -> Path | None:
    root = os.environ.get("GENIE_G1_OFFICIAL_DIR", str(Path.home() / "genie_g1_official_hf"))
    for rel in (
        "robot/G1_omnipicker/robot.usd",
        "robot/G1_120s/robot.usda",
    ):
        p = Path(root).expanduser() / rel
        if p.is_file():
            return p.resolve()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Load Genie G1 USD in Isaac Sim (standalone)")
    ap.add_argument("--usd", type=Path, default=None, help="Path to root USD (default: GENIE_G1_OFFICIAL_DIR + G1_omnipicker)")
    ap.add_argument("--headless", action="store_true", help="Run without GUI window")
    ap.add_argument("--steps", type=int, default=120, help="Physics steps before exit (smoke test)")
    args = ap.parse_args()

    usd = args.usd
    if usd is None:
        cand = _default_usd()
        if cand is None:
            print(
                "No --usd and no file under GENIE_G1_OFFICIAL_DIR. Run:\n"
                "  bash ../mujoco_g1_infer/fetch_official_genie_g1_sim_assets.sh",
                file=sys.stderr,
            )
            return 1
        usd = cand
    else:
        usd = Path(usd).expanduser().resolve()
    if not usd.is_file():
        print(f"USD not found: {usd}", file=sys.stderr)
        return 1

    try:
        from isaacsim import SimulationApp
    except ImportError:
        print(
            "Import isaacsim failed. Run this script via Isaac Sim's interpreter, e.g.\n"
            "  ./run_with_isaac_python.sh load_genie_g1_usd_standalone.py --usd ...",
            file=sys.stderr,
        )
        return 2

    app = SimulationApp({"headless": bool(args.headless)})
    try:
        import omni.usd
        from pxr import UsdGeom

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("Failed to create USD stage", file=sys.stderr)
            return 3

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        g1 = UsdGeom.Xform.Define(stage, "/World/G1")
        g1.GetPrim().GetReferences().AddReference(str(usd))

        # One warm-up tick so extensions settle (version-dependent).
        for _ in range(max(1, int(args.steps))):
            app.update()
        print("OK loaded:", usd, "steps=", args.steps)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
