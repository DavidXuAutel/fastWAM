#!/usr/bin/env python3
"""Prepare Apex-WAM-Mini training data per docs/Apex-WAM-Mini-Design-v3.2.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from manifest import (  # noqa: E402
    build_stage_manifest,
    load_sources_config,
    scan_sources,
    write_manifest,
    write_scan_report,
)
from registry import CompatibilityRegistry  # noqa: E402
from verify_source import verify_lerobot_source, write_verification_artifact  # noqa: E402


def _resolve(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _active_profile(sources_cfg: Dict[str, Any], override: Optional[str]) -> str:
    return override or sources_cfg.get("active_profile", "g1")


def cmd_scan(args: argparse.Namespace) -> int:
    sources_path = _resolve(Path(args.sources))
    sources_cfg = load_sources_config(sources_path)
    profile = _active_profile(sources_cfg, args.profile)
    scan = scan_sources(sources_cfg)
    out = _resolve(Path(args.out))
    write_scan_report(scan, out)

    print(f"Wrote scan report: {out} (profile={profile})")
    for stat in scan:
        status = "OK" if stat.exists and stat.error is None else "MISSING"
        hours = f"{stat.hours:.2f}h" if stat.hours else "n/a"
        print(f"  [{status}] {stat.source_id}: {stat.lerobot_root} ({hours})")
        if stat.error:
            print(f"         error: {stat.error}")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    sources_path = _resolve(Path(args.sources))
    sources_cfg = load_sources_config(sources_path)
    scan = scan_sources(sources_cfg)
    manifest = build_stage_manifest(args.stage, sources_cfg, scan, profile=args.profile)
    out = _resolve(Path(args.out))
    write_manifest(manifest, out)
    print(f"Wrote stage {manifest.stage} manifest: {out} (profile={manifest.totals.get('profile')})")
    print(f"  weighted_hours_estimate: {manifest.totals.get('weighted_hours_estimate')}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    registry_path = _resolve(Path(args.registry))
    sources_path = _resolve(Path(args.sources))
    registry = CompatibilityRegistry.load(registry_path, profile=args.profile)
    sources_cfg = load_sources_config(sources_path)
    artifact_dir = _resolve(Path(args.artifact_dir))

    results = []
    for source_id, entry in sources_cfg.get("sources", {}).items():
        spec = registry.lookup(source_id)
        if spec is None:
            continue
        roots = entry.get("lerobot_roots", [])
        if not roots:
            continue
        root = _resolve(Path(roots[0]))
        result = verify_lerobot_source(
            source_id=source_id,
            lerobot_root=root,
            registry=registry,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        path = write_verification_artifact(result, artifact_dir)
        results.append(result)
        print(f"  {source_id}: passed={result.passed} artifact={path}")
        if args.mark_verified and result.passed:
            registry.save_verified_flag(registry_path, source_id, True)
            print(f"    -> marked verified=true in {registry_path}")

    summary = {
        "passed": [r.source_id for r in results if r.passed],
        "failed": [r.source_id for r in results if not r.passed],
    }
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verification summary: {summary_path}")
    return 0 if not summary["failed"] else 1


def cmd_all(args: argparse.Namespace) -> int:
    data_root = _resolve(Path(args.data_root))
    reports = data_root / "reports"
    manifests = data_root / "manifests"
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    scan_args = argparse.Namespace(
        sources=args.sources,
        out=str(reports / "dataset_scan.json"),
        profile=args.profile,
    )
    cmd_scan(scan_args)

    profiles = [args.profile] if args.profile else ["g1", "franka"]
    for prof in profiles:
        for stage in ("B", "C"):
            manifest_args = argparse.Namespace(
                sources=args.sources,
                stage=stage,
                profile=prof,
                out=str(manifests / f"stage_{stage.lower()}_{prof}.json"),
            )
            cmd_manifest(manifest_args)

    if args.verify:
        for prof in profiles:
            verify_args = argparse.Namespace(
                registry=args.registry,
                sources=args.sources,
                artifact_dir=str(reports / f"registry_verification_{prof}"),
                sample_size=args.sample_size,
                seed=args.seed,
                mark_verified=args.mark_verified,
                profile=prof,
            )
            rc = cmd_verify(verify_args)
            if rc != 0:
                return rc
        return 0

    print("\nNext steps:")
    print("  1. Place LeRobot datasets under data/apex_wam_mini/ (see data/apex_wam_mini/README.md)")
    print("  2. Edit experiments/apex_wam_mini/sources.yaml paths")
    print("  3. Run: python experiments/apex_wam_mini/prepare_data.py verify --mark-verified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sources",
        default="experiments/apex_wam_mini/sources.yaml",
        help="Local source path config",
    )
    p.add_argument(
        "--registry",
        default="configs/data_compatibility.yaml",
        help="Compatibility registry",
    )
    p.add_argument(
        "--profile",
        choices=["g1", "franka"],
        default=None,
        help="Target robot profile (default: sources.yaml active_profile; 'all' generates both)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--sources", default="experiments/apex_wam_mini/sources.yaml")
        sp.add_argument("--registry", default="configs/data_compatibility.yaml")
        sp.add_argument("--profile", choices=["g1", "franka"], default=None)

    scan_p = sub.add_parser("scan", help="Scan LeRobot roots and report hours/frames")
    add_common(scan_p)
    scan_p.add_argument("--out", default="data/apex_wam_mini/reports/dataset_scan.json")
    scan_p.set_defaults(func=cmd_scan)

    man_p = sub.add_parser("manifest", help="Build stage B/C sampling manifest")
    add_common(man_p)
    man_p.add_argument("--stage", choices=["B", "C"], required=True)
    man_p.add_argument("--out", default="data/apex_wam_mini/manifests/stage_b_g1.json")
    man_p.set_defaults(func=cmd_manifest)

    ver_p = sub.add_parser("verify", help="Verify compatible sources against registry criteria")
    add_common(ver_p)
    ver_p.add_argument("--artifact-dir", default="data/apex_wam_mini/reports/registry_verification")
    ver_p.add_argument("--sample-size", type=int, default=100)
    ver_p.add_argument("--seed", type=int, default=42)
    ver_p.add_argument(
        "--mark-verified",
        action="store_true",
        help="Set verified=true in registry for sources that pass",
    )
    ver_p.set_defaults(func=cmd_verify)

    all_p = sub.add_parser("all", help="Scan + manifest B/C for g1+franka (+ optional verify)")
    add_common(all_p)
    all_p.add_argument("--data-root", default="data/apex_wam_mini")
    all_p.add_argument("--verify", action="store_true")
    all_p.add_argument("--sample-size", type=int, default=100)
    all_p.add_argument("--seed", type=int, default=42)
    all_p.add_argument("--mark-verified", action="store_true")
    all_p.set_defaults(func=cmd_all)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
