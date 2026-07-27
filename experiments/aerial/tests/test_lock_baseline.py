import json

import pytest

from experiments.aerial.eval.lock_baseline import (
    build_lock_manifest,
    main,
    select_baseline,
)


def test_select_lowest_ne_tie_breaks_later_step():
    candidates = [
        {"step": 1000, "mean_ne": 150.0, "checkpoint": "a", "metrics_path": "a.json", "sha256": "1"},
        {"step": 4000, "mean_ne": 120.0, "checkpoint": "b", "metrics_path": "b.json", "sha256": "2"},
        {"step": 5000, "mean_ne": 120.0, "checkpoint": "c", "metrics_path": "c.json", "sha256": "3"},
    ]
    chosen = select_baseline(candidates)
    assert chosen["step"] == 5000
    man = build_lock_manifest(
        chosen,
        candidates=candidates,
        stamp="20260727-072347-5k-2gpu-b0-to-joint-video",
    )
    assert man["s1_ne"] == 96.0
    assert man["baseline_mean_ne"] == 120.0


def test_select_baseline_rejects_no_finite_candidates():
    candidates = [
        {"step": 1000, "mean_ne": float("inf"), "checkpoint": "a", "metrics_path": "a.json", "sha256": "1"},
        {"step": 2000, "mean_ne": float("nan"), "checkpoint": "b", "metrics_path": "b.json", "sha256": "2"},
    ]
    with pytest.raises(ValueError, match="no finite mean_ne candidates"):
        select_baseline(candidates)


def test_select_baseline_ignores_non_finite():
    candidates = [
        {"step": 1000, "mean_ne": float("inf"), "checkpoint": "a", "metrics_path": "a.json", "sha256": "1"},
        {"step": 2000, "mean_ne": 100.0, "checkpoint": "b", "metrics_path": "b.json", "sha256": "2"},
    ]
    chosen = select_baseline(candidates)
    assert chosen["step"] == 2000


def test_cli_writes_baseline_lock_manifest(tmp_path):
    metrics_a = tmp_path / "a.json"
    metrics_a.write_text('{"NE": 150.0, "n": 20.0}\n')
    metrics_b = tmp_path / "b.json"
    metrics_b.write_text('{"NE": 120.0, "n": 20.0}\n')
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    ckpt_c = tmp_path / "c.pt"
    out = tmp_path / "baseline_lock.manifest.json"

    rc = main(
        [
            "--stamp",
            "test-stamp",
            "--candidate",
            f"1000={ckpt_a}={metrics_a}=sha1",
            "--candidate",
            f"4000={ckpt_b}={metrics_b}=sha2",
            "--candidate",
            f"5000={ckpt_c}={metrics_b}=sha3",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["checkpoint"] == str(ckpt_c)
    assert man["s1_ne"] == 96.0
    assert man["selection_rule"] == "min_mean_ne_tie_later_step"
