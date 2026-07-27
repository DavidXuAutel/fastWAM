from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.aerial.orchestration.checkpoint import is_complete_checkpoint
from experiments.aerial.orchestration.eval_queue import enqueue

B1_STEPS = (250, 500, 1000)


def default_b1_metrics_path(results_root: Path, stamp: str, step: int) -> Path:
    return results_root / f"b1_{stamp}" / f"step_{step:06d}_seen20" / "metrics.json"


def _ensure_sha256_sidecar(pt: Path) -> None:
    sidecar = Path(str(pt) + ".sha256")
    if sidecar.is_file():
        return
    digest = hashlib.sha256()
    with pt.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    sidecar.write_text(digest.hexdigest() + f"  {pt.name}\n", encoding="utf-8")


def discover_b1_checkpoints(
    weights_dir: Path,
    *,
    steps: Sequence[int] = B1_STEPS,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for step in steps:
        pt = weights_dir / f"step_{int(step):06d}.pt"
        if not pt.is_file():
            continue
        try:
            size1 = pt.stat().st_size
        except FileNotFoundError:
            continue
        if size1 < min_bytes:
            continue
        if settle_s > 0:
            time.sleep(settle_s)
            try:
                if pt.stat().st_size != size1:
                    continue
            except FileNotFoundError:
                continue
        _ensure_sha256_sidecar(pt)
        if not is_complete_checkpoint(pt, settle_s=0.0, min_bytes=min_bytes):
            continue
        found.append(
            {
                "step": int(step),
                "checkpoint": str(pt.resolve()),
                "sha256_path": str(Path(str(pt) + ".sha256").resolve()),
            }
        )
    return found


def build_b1_eval_job(
    *,
    stamp: str,
    step: int,
    checkpoint: str,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
) -> dict[str, Any]:
    out_metrics = default_b1_metrics_path(results_root, stamp, step)
    return {
        "id": f"b1-{stamp}-step_{step:06d}",
        "kind": "b1",
        "checkpoint": checkpoint,
        "out_metrics": str(out_metrics),
        "task": task,
        "ann": str(ann),
        "openfly_root": str(openfly_root),
        "seed": 42,
        "max_steps": 100,
        "max_episodes": 20,
    }


def enqueue_ready_b1_jobs(
    *,
    stamp: str,
    weights_dir: Path,
    queue_dir: Path,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
    steps: Sequence[int] = B1_STEPS,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for item in discover_b1_checkpoints(
        weights_dir, steps=steps, min_bytes=min_bytes, settle_s=settle_s
    ):
        job = build_b1_eval_job(
            stamp=stamp,
            step=int(item["step"]),
            checkpoint=str(item["checkpoint"]),
            results_root=results_root,
            ann=ann,
            openfly_root=openfly_root,
            task=task,
        )
        Path(job["out_metrics"]).parent.mkdir(parents=True, exist_ok=True)
        enqueue(queue_dir, job)
        jobs.append(job)
    return jobs


def watch_and_enqueue(
    *,
    stamp: str,
    weights_dir: Path,
    queue_dir: Path,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
    steps: Sequence[int] = B1_STEPS,
    poll_s: float = 60.0,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
    once: bool = False,
) -> None:
    while True:
        jobs = enqueue_ready_b1_jobs(
            stamp=stamp,
            weights_dir=weights_dir,
            queue_dir=queue_dir,
            results_root=results_root,
            ann=ann,
            openfly_root=openfly_root,
            task=task,
            steps=steps,
            min_bytes=min_bytes,
            settle_s=settle_s,
        )
        print(f"enqueued={len(jobs)}")
        for job in jobs:
            print(job["id"], job["checkpoint"], job["out_metrics"])
        if once:
            return
        time.sleep(poll_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch B1 checkpoints and enqueue evals")
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--openfly-root", type=Path, required=True)
    parser.add_argument("--task", default="aerial_joint_b1_joint")
    parser.add_argument("--steps", default="250,500,1000")
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--min-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--settle-s", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    steps = tuple(int(p.strip()) for p in args.steps.split(",") if p.strip())
    watch_and_enqueue(
        stamp=args.stamp,
        weights_dir=args.weights_dir,
        queue_dir=args.queue_dir,
        results_root=args.results_root,
        ann=args.ann,
        openfly_root=args.openfly_root,
        task=args.task,
        steps=steps,
        poll_s=args.poll_s,
        min_bytes=args.min_bytes,
        settle_s=args.settle_s,
        once=args.once,
    )


if __name__ == "__main__":
    main()
