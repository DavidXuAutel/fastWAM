from __future__ import annotations

from pathlib import Path

from experiments.aerial.orchestration.b1_discover import (
    B1_STEPS,
    build_b1_eval_job,
    enqueue_ready_b1_jobs,
)


def test_build_b1_eval_job_paths():
    job = build_b1_eval_job(
        stamp="s1",
        step=250,
        checkpoint="/tmp/step_000250.pt",
        results_root=Path("/tmp/results"),
        ann=Path("/tmp/ann.json"),
        openfly_root=Path("/tmp/openfly"),
    )
    assert job["id"] == "b1-s1-step_000250"
    assert job["kind"] == "b1"
    assert job["out_metrics"].endswith("b1_s1/step_000250_seen20/metrics.json")
    assert job["seed"] == 42
    assert job["max_episodes"] == 20


def test_enqueue_ready_b1_jobs(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    queue = tmp_path / "queue"
    results = tmp_path / "results"
    ann = tmp_path / "ann.json"
    ann.write_text("[]\n", encoding="utf-8")
    openfly = tmp_path / "openfly"
    openfly.mkdir()
    for step in B1_STEPS:
        pt = weights / f"step_{step:06d}.pt"
        pt.write_bytes(b"x" * 100)
        (Path(str(pt) + ".sha256")).write_text("abcd\n", encoding="utf-8")
    jobs = enqueue_ready_b1_jobs(
        stamp="s1",
        weights_dir=weights,
        queue_dir=queue,
        results_root=results,
        ann=ann,
        openfly_root=openfly,
        min_bytes=10,
        settle_s=0.0,
    )
    assert len(jobs) == 3
    assert (queue / "pending").is_dir()
    assert len(list((queue / "pending").glob("*.json"))) == 3
