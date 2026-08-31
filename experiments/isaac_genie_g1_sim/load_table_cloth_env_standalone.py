#!/usr/bin/env python3
"""
Open the table+cloth experiment USD in Isaac Sim (SimulationApp + physics ticks).

  ./run_with_isaac_python.sh load_table_cloth_env_standalone.py --usd ~/isaac_sim_exports/table_cloth_env.usd

Or headless smoke:
  ... --headless --steps 60
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _default_usd() -> Path:
    er = os.environ.get("ISAAC_SIM_EXPORT_ROOT")
    if er:
        return Path(er).expanduser() / "table_cloth_env.usd"
    if os.environ.get("ISAAC_DOCKER_DATA_ROOT"):
        dr = Path(os.environ["ISAAC_DOCKER_DATA_ROOT"]).expanduser()
        return dr / "data" / "genie_sim" / "table_cloth_env.usd"
    return Path.home() / "isaac_sim_exports" / "table_cloth_env.usd"


def main() -> int:
    ap = argparse.ArgumentParser(description="Load table_cloth_env.usd in Isaac Sim")
    ap.add_argument("--usd", type=Path, default=None, help="Path to USD (default: ISAAC_SIM_EXPORT_ROOT or ~/isaac_sim_exports/table_cloth_env.usd)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=120, help="Simulation ticks after load (ignored with --interactive)")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Keep SimulationApp running until window closed (GUI only; use without --headless)",
    )
    args = ap.parse_args()

    usd = args.usd
    if usd is None:
        usd = _default_usd()
    else:
        usd = Path(usd).expanduser().resolve()
    if not usd.is_file():
        print(f"USD not found: {usd}\nRun first: bash ~/isaac_genie_g1_sim/run_build_table_cloth_env_on_host.sh", file=sys.stderr)
        return 1

    try:
        from isaacsim import SimulationApp
    except ImportError:
        print("Run via Isaac python.sh + run_with_isaac_python.sh", file=sys.stderr)
        return 2

    app = SimulationApp({"headless": bool(args.headless)})
    try:
        import omni.usd
        from pxr import UsdGeom

        omni.usd.get_context().open_stage(str(usd))
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("open_stage failed", file=sys.stderr)
            return 3
        world = stage.GetPrimAtPath("/World")
        if world and world.IsValid():
            stage.SetDefaultPrim(world)
        else:
            d = stage.GetDefaultPrim()
            if not d or not d.IsValid():
                xf = UsdGeom.Xform.Define(stage, "/World")
                stage.SetDefaultPrim(xf.GetPrim())

        if bool(args.interactive) and not bool(args.headless):
            is_running_fn = getattr(app, "is_running", None)
            if callable(is_running_fn):
                while is_running_fn():
                    app.update()
            else:
                print(
                    "WARN: SimulationApp has no is_running(); falling back to --steps.",
                    file=sys.stderr,
                )
                for _ in range(max(1, int(args.steps))):
                    app.update()
        else:
            for _ in range(max(1, int(args.steps))):
                app.update()
        print("OK loaded stage:", usd, "interactive=", args.interactive, "steps=", args.steps)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
