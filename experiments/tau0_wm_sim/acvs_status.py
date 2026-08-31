#!/usr/bin/env python3
"""Inspect readiness for tau0-WM ACVS/TTC upgrade."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _path_check(path: Path, *, expect_dir: Optional[bool] = None) -> Dict[str, Any]:
    exists = path.exists()
    ok = exists
    if exists and expect_dir is not None:
        ok = path.is_dir() if expect_dir else path.is_file()
    return {"ok": bool(ok), "path": str(path), "exists": bool(exists)}


def inspect_acvs_readiness(*, tau_repo: Path, simulator_checkpoint: Path) -> Dict[str, Any]:
    tau_repo = tau_repo.expanduser().resolve()
    simulator_checkpoint = simulator_checkpoint.expanduser().resolve()
    checks = {
        "tau_repo": _path_check(tau_repo, expect_dir=True),
        "simulator_checkpoint": _path_check(simulator_checkpoint),
        "ttc_code": _path_check(tau_repo / "web_infer_utils" / "test_time_compute.py", expect_dir=False),
        "acvs_client": _path_check(tau_repo / "web_infer_utils" / "AcvsPolicy.py", expect_dir=False),
        "simulator_config": _path_check(tau_repo / "configs" / "deployment" / "acvs.yaml", expect_dir=False),
    }
    missing = [name for name, check in checks.items() if not check["ok"]]
    return {
        "ready": not missing,
        "checks": checks,
        "missing": missing,
        "note": (
            "The public tau0-WM README stated simulator weights and TTC code would be released later; "
            "this check becomes ready once those assets are present or local replacements are provided."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-repo", type=Path, required=True)
    parser.add_argument("--simulator-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_acvs_readiness(tau_repo=args.tau_repo, simulator_checkpoint=args.simulator_checkpoint)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
