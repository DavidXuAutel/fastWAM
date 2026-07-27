from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from experiments.aerial.orchestration.state import write_status


def metrics_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ne = float(data.get("NE", "nan"))
        n = float(data.get("n", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return math.isfinite(ne) and n >= 1


def _job_path(queue_dir: Path, subdir: str, job_id: str) -> Path:
    return queue_dir / subdir / f"{job_id}.json"


def _read_job(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _next_enqueue_seq(queue_dir: Path) -> int:
    queue_dir.mkdir(parents=True, exist_ok=True)
    seq_path = queue_dir / ".enqueue_seq"
    if seq_path.is_file():
        seq = int(seq_path.read_text(encoding="utf-8").strip()) + 1
    else:
        seq = 1
    fd, tmp = tempfile.mkstemp(dir=str(queue_dir), prefix=".enqueue_seq.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{seq}\n")
        os.replace(tmp, seq_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return seq


def _pending_jobs(queue_dir: Path) -> list[tuple[int, Path, dict[str, Any]]]:
    pending_dir = queue_dir / "pending"
    if not pending_dir.is_dir():
        return []
    jobs: list[tuple[int, Path, dict[str, Any]]] = []
    for path in pending_dir.glob("*.json"):
        job = _read_job(path)
        jobs.append((int(job["_enqueued_at"]), path, job))
    jobs.sort(key=lambda item: item[0])
    return jobs


def _move_to_done(queue_dir: Path, pending_path: Path, job_id: str) -> None:
    done_path = _job_path(queue_dir, "done", job_id)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending_path, done_path)


def enqueue(queue_dir: Path, job: dict[str, Any]) -> str:
    job_id = str(job["id"])
    for subdir in ("pending", "running", "done"):
        if _job_path(queue_dir, subdir, job_id).is_file():
            return job_id
    record = dict(job)
    record["_enqueued_at"] = _next_enqueue_seq(queue_dir)
    path = _job_path(queue_dir, "pending", job_id)
    write_status(path, record)
    return job_id


def claim_next(queue_dir: Path) -> dict[str, Any] | None:
    for _seq, path, job in _pending_jobs(queue_dir):
        out_metrics = Path(str(job["out_metrics"]))
        if metrics_valid(out_metrics):
            _move_to_done(queue_dir, path, str(job["id"]))
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
