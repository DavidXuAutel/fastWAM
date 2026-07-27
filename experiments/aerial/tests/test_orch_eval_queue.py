from experiments.aerial.orchestration.eval_queue import enqueue, claim_next, mark_done


def test_fifo_claim_and_skip_existing_metrics(tmp_path):
    q = tmp_path / "queue"
    metrics = tmp_path / "m.json"
    metrics.write_text('{"NE": 1.0, "SR": 0.0, "n": 20}\n')
    jid = enqueue(q, {
        "id": "b0-step_001000",
        "kind": "b0",
        "checkpoint": "/c.pt",
        "out_metrics": str(metrics),
        "task": "aerial_joint_b1_joint",
        "ann": "/a.json",
        "openfly_root": "/of",
        "seed": 42,
        "max_steps": 100,
        "max_episodes": 20,
    })
    assert claim_next(q) is None  # already has valid metrics
    metrics.unlink()
    job = claim_next(q)
    assert job is not None and job["id"] == jid
    mark_done(q, jid, {"NE": 12.3, "SR": 0.0, "n": 20.0})
    assert claim_next(q) is None
