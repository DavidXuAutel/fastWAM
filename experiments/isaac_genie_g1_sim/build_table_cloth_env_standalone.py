#!/usr/bin/env python3
"""
Programmatic Isaac Sim scene: physics scene, dome-style background light, static floor,
kinematic table (rigid body + collision), optional PhysX particle cloth (grid mesh).

Run only with Isaac's python.sh, e.g.:
  ./run_with_isaac_python.sh build_table_cloth_env_standalone.py --out ~/table_cloth_env.usd
  # headless server:
  ./run_with_isaac_python.sh build_table_cloth_env_standalone.py --headless --out /path/out.usd

Cloth uses PhysxParticleClothAPI + PhysxAutoParticleClothAPI + PhysxParticleSystem when
pxr.PhysxSchema is available; otherwise the cloth mesh is still authored for manual upgrade in GUI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _grid_cloth_mesh(*, nx: int, ny: int, sx: float, sy: float, z: float):
    """Triangle mesh for a rectangular cloth sheet in XY at height z."""
    from pxr import Gf

    pts: list[Gf.Vec3f] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = (i / max(nx, 1) - 0.5) * sx
            y = (j / max(ny, 1) - 0.5) * sy
            pts.append(Gf.Vec3f(x, y, z))
    idx: list[int] = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1) + 1
            v3 = v0 + (nx + 1)
            idx.extend([v0, v1, v2, v0, v2, v3])
    ntri = len(idx) // 3
    counts = [3] * ntri
    return pts, idx, counts


def _build_scene(stage, *, cloth_res: int, sim_steps: int, skip_cloth_physics: bool) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    UsdPhysics.Scene.Define(stage, "/World/physicsScene")

    # Background lighting (studio-style dome; swap texture via GUI or add texture path later).
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1200.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.86, 0.92))
    dome.AddRotateXOp().Set(0.0)

    # Static floor (collision only, no rigid body).
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    ground.AddScaleOp().Set(Gf.Vec3d(24.0, 24.0, 0.02))
    ground.CreateSizeAttr(1.0)
    ground_prim = ground.GetPrim()
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    UsdGeom.Gprim(ground_prim).CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.38)])

    # Kinematic table (stable furniture).
    table = UsdGeom.Xform.Define(stage, "/World/Table")
    table.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.78))
    table_prim = table.GetPrim()
    rb = UsdPhysics.RigidBodyAPI.Apply(table_prim)
    rb.CreateKinematicEnabledAttr(True)
    top = UsdGeom.Cube.Define(stage, "/World/Table/Top")
    top.CreateSizeAttr(1.0)
    top.AddScaleOp().Set(Gf.Vec3d(1.15, 0.65, 0.045))
    top.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    top_prim = top.GetPrim()
    UsdPhysics.CollisionAPI.Apply(top_prim)
    UsdGeom.Gprim(top_prim).CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.28, 0.12)])

    # Cloth mesh (grid).
    nx = ny = max(8, int(cloth_res))
    pts, indices, counts = _grid_cloth_mesh(nx=nx, ny=ny, sx=0.55, sy=0.42, z=1.05)
    cloth = UsdGeom.Mesh.Define(stage, "/World/Cloth")
    cloth.CreatePointsAttr(pts)
    cloth.CreateFaceVertexIndicesAttr(indices)
    cloth.CreateFaceVertexCountsAttr(counts)
    cloth.CreateDoubleSidedAttr(True)
    UsdGeom.Gprim(cloth.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.2, 0.25)])

    if skip_cloth_physics:
        print("INFO: --skip-cloth-physics: cloth is visual mesh only; add particle cloth in GUI if needed.")
        return

    try:
        from pxr import PhysxSchema
    except ImportError:
        print("WARN: pxr.PhysxSchema not available; cloth left as plain mesh.", file=sys.stderr)
        return

    try:
        ps_path = "/World/ParticleSystem"
        PhysxSchema.PhysxParticleSystem.Define(stage, ps_path)
        cloth_prim = cloth.GetPrim()
        p_cloth = PhysxSchema.PhysxParticleClothAPI.Apply(cloth_prim)
        rel = p_cloth.CreateParticleSystemRel()
        rel.SetTargets([Sdf.Path(ps_path)])
        PhysxSchema.PhysxAutoParticleClothAPI.Apply(cloth_prim)
        print("OK: PhysX particle cloth APIs applied to /World/Cloth ->", ps_path)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: cloth physics setup failed ({exc}); mesh kept without particle cloth.", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="Output USD path (default: ~/isaac_sim_exports/table_cloth_env.usd)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=180, help="Simulation warm-up ticks before export")
    ap.add_argument("--cloth-res", type=int, default=16, help="Grid resolution for cloth mesh (per axis)")
    ap.add_argument("--skip-cloth-physics", action="store_true", help="Only static + table; no PhysX cloth APIs")
    args = ap.parse_args()

    out = args.out
    if out is None:
        out = Path.home() / "isaac_sim_exports" / "table_cloth_env.usd"
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from isaacsim import SimulationApp
    except ImportError:
        print(
            "Import isaacsim failed. Run via Isaac python.sh, e.g.\n"
            "  ./run_with_isaac_python.sh build_table_cloth_env_standalone.py --out ...",
            file=sys.stderr,
        )
        return 2

    app = SimulationApp({"headless": bool(args.headless)})
    try:
        import omni.usd

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("Failed to create stage", file=sys.stderr)
            return 3

        _build_scene(stage, cloth_res=args.cloth_res, sim_steps=args.steps, skip_cloth_physics=args.skip_cloth_physics)

        for _ in range(max(1, int(args.steps))):
            app.update()

        stage.GetRootLayer().Export(str(out))
        print("OK exported:", out)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
