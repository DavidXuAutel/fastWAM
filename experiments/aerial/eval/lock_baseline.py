from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

from experiments.aerial.orchestration.state import write_status


def select_baseline(candidates: list[dict]) -> dict:
    finite = [c for c in candidates if math.isfinite(float(c["mean_ne"]))]
    if not finite:
        raise ValueError("no finite mean_ne candidates")
    return sorted(finite, key=lambda c: (float(c["mean_ne"]), -int(c["step"])))[0]


def build_lock_manifest(chosen: dict, *, candidates: list[dict], stamp: str) -> dict:
    baseline = float(chosen["mean_ne"])
    return {
        "stamp": stamp,
        "checkpoint": chosen["checkpoint"],
        "sha256": chosen["sha256"],
        "metrics_path": chosen["metrics_path"],
        "baseline_mean_ne": baseline,
        "s1_ne": 0.8 * baseline,
        "candidates": candidates,
        "selection_rule": "min_mean_ne_tie_later_step",
    }


def read_mean_ne(metrics_path: Path) -> float:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(data["NE"])


def parse_candidate(spec: str) -> dict:
    parts = spec.split("=", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"invalid candidate spec: {spec!r}")
    step_str, checkpoint, metrics_path, sha256 = parts
    metrics = Path(metrics_path)
    return {
        "step": int(step_str),
        "checkpoint": checkpoint,
        "metrics_path": str(metrics),
        "mean_ne": read_mean_ne(metrics),
        "sha256": sha256,
    }


def write_lock_manifest(path: Path, manifest: dict) -> None:
    write_status(path, manifest)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock B0 baseline from eval candidates")
    parser.add_argument("--stamp", required=True)
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        metavar="STEP=CKPT=METRICS=SHA256",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    candidates = [parse_candidate(spec) for spec in args.candidate]
    chosen = select_baseline(candidates)
    manifest = build_lock_manifest(chosen, candidates=candidates, stamp=args.stamp)
    write_lock_manifest(args.out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
