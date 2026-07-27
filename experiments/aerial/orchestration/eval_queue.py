from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from experiments.aerial.orchestration.state import write_status


def metrics_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return math.isfinite(float(data.get("NE", "nan"))) and float(data.get("n", 0)) >= 1


def _job_path(queue_dir: Path, subdir: str, job_id: str) -> Path:
    return queue_dir / subdir / f"{job_id}.json"


def _read_job(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def enqueue(queue_dir: Path, job: dict[str, Any]) -> str:
    job_id = str(job["id"])
    for subdir in ("pending", "running", "done"):
        if _job_path(queue_dir, subdir, job_id).is_file():
            return job_id
    path = _job_path(queue_dir, "pending", job_id)
    write_status(path, job)
    return job_id


def claim_next(queue_dir: Path) -> dict[str, Any] | None:
    pending_dir = queue_dir / "pending"
    if not pending_dir.is_dir():
        return None
    for path in sorted(pending_dir.glob("*.json")):
        job = _read_job(path)
        out_metrics = Path(str(job["out_metrics"]))
        if metrics_valid(out_metrics):
            continue
        running_path = _job_path(queue_dir, "running", job["id"])
        running_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, running_path)
        return job
    return None


def mark_done(queue_dir: Path, job_id: str, result: dict[str, Any]) -> None:
    running_path = _job_path(queue_dir, "running", job_id)
    job = _read_job(running_path)
    write_status(Path(str(job["out_metrics"])), result)
    done_path = _job_path(queue_dir, "done", job_id)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(running_path, done_path)
