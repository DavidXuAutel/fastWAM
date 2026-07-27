from __future__ import annotations

import json
from datetime import datetime

import pytest

from experiments.aerial.eval.lock_baseline import (
    ORCHESTRATION_VERSION,
    build_lock_manifest,
    load_metrics,
    main,
    parse_candidate_json,
    parse_candidate_legacy,
    select_baseline,
    validate_candidates_unique,
    write_lock_manifest,
)
from experiments.aerial.orchestration.state import read_status


def _sha(tag: str = "a") -> str:
    base = "".join(ch for ch in tag.lower() if ch in "0123456789abcdef") or "a"
    return (base * 64)[:64]


def _candidate(
    step: int,
    mean_ne: float,
    checkpoint: str = "ckpt",
    metrics_path: str = "metrics.json",
    sha256: str | None = None,
) -> dict:
    return {
        "step": step,
        "mean_ne": mean_ne,
        "checkpoint": checkpoint,
        "metrics_path": metrics_path,
        "sha256": sha256 or _sha(str(step)),
    }


def test_select_lowest_ne_tie_breaks_later_step():
    candidates = [
        _candidate(1000, 150.0, checkpoint="a", metrics_path="a.json", sha256=_sha("1")),
        _candidate(4000, 120.0, checkpoint="b", metrics_path="b.json", sha256=_sha("2")),
        _candidate(5000, 120.0, checkpoint="c", metrics_path="c.json", sha256=_sha("3")),
    ]
    chosen = select_baseline(candidates)
    assert chosen["step"] == 5000
    man = build_lock_manifest(
        chosen,
        candidates=candidates,
        stamp="20260727-072347-5k-2gpu-b0-to-joint-video",
        selection_time="2026-07-27T12:00:00+00:00",
    )
    assert man["s1_ne"] == 96.0
    assert man["baseline_mean_ne"] == 120.0
    assert man["selection_time"] == "2026-07-27T12:00:00+00:00"
    assert man["orchestration_version"] == ORCHESTRATION_VERSION


def test_select_baseline_rejects_no_finite_candidates():
    candidates = [
        _candidate(1000, float("inf")),
        _candidate(2000, float("nan")),
    ]
    with pytest.raises(ValueError, match="non-finite mean_ne"):
        select_baseline(candidates)


def test_select_baseline_rejects_any_non_finite():
    candidates = [
        _candidate(1000, float("inf")),
        _candidate(2000, 100.0),
    ]
    with pytest.raises(ValueError, match="non-finite mean_ne"):
        select_baseline(candidates)


def test_select_baseline_rejects_missing_mean_ne():
    candidates = [{"step": 1000, "checkpoint": "a", "metrics_path": "a.json", "sha256": _sha()}]
    with pytest.raises(ValueError, match="missing mean_ne"):
        select_baseline(candidates)


def test_load_metrics_requires_finite_ne_and_n(tmp_path):
    good = tmp_path / "good.json"
    good.write_text('{"NE": 100.0, "n": 20.0}\n')
    assert load_metrics(good) == 100.0

    bad_n = tmp_path / "bad_n.json"
    bad_n.write_text('{"NE": 100.0, "n": 0}\n')
    with pytest.raises(ValueError, match="invalid n"):
        load_metrics(bad_n)

    bad_ne = tmp_path / "bad_ne.json"
    bad_ne.write_text('{"NE": "nan", "n": 20.0}\n')
    with pytest.raises(ValueError, match="non-finite NE"):
        load_metrics(bad_ne)

    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json\n")
    with pytest.raises(ValueError, match="invalid metrics JSON"):
        load_metrics(malformed)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="metrics file not found"):
        load_metrics(missing)


def test_validate_candidates_unique_rejects_duplicates():
    base = _candidate(1000, 100.0, checkpoint="/a.pt", metrics_path="/a.json")
    other = _candidate(2000, 110.0, checkpoint="/b.pt", metrics_path="/b.json")

    with pytest.raises(ValueError, match="duplicate step"):
        validate_candidates_unique([base, base])

    dup_ckpt = _candidate(2000, 110.0, checkpoint="/a.pt", metrics_path="/b.json")
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        validate_candidates_unique([base, dup_ckpt])

    dup_metrics = _candidate(2000, 110.0, checkpoint="/b.pt", metrics_path="/a.json")
    with pytest.raises(ValueError, match="duplicate metrics_path"):
        validate_candidates_unique([base, dup_metrics])

    validate_candidates_unique([base, other])


def test_build_lock_manifest_rejects_chosen_not_in_candidates():
    chosen = _candidate(5000, 120.0)
    candidates = [_candidate(1000, 150.0), _candidate(4000, 130.0)]
    with pytest.raises(ValueError, match="chosen candidate not in candidates"):
        build_lock_manifest(
            chosen,
            candidates=candidates,
            stamp="x",
            selection_time="2026-07-27T12:00:00+00:00",
        )


def test_parse_candidate_json_validates_paths_and_sha(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"weights")
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"NE": 120.0, "n": 20.0}\n')

    parsed = parse_candidate_json(
        json.dumps(
            {
                "step": 5000,
                "checkpoint": str(ckpt),
                "metrics_path": str(metrics),
                "sha256": _sha("c"),
            }
        )
    )
    assert parsed["step"] == 5000
    assert parsed["mean_ne"] == 120.0

    with pytest.raises(ValueError, match="invalid sha256"):
        parse_candidate_json(
            json.dumps(
                {
                    "step": 5000,
                    "checkpoint": str(ckpt),
                    "metrics_path": str(metrics),
                    "sha256": "tooshort",
                }
            )
        )

    missing_ckpt = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="checkpoint not found"):
        parse_candidate_json(
            json.dumps(
                {
                    "step": 5000,
                    "checkpoint": str(missing_ckpt),
                    "metrics_path": str(metrics),
                    "sha256": _sha("c"),
                }
            )
        )


def test_parse_candidate_json_accepts_equals_in_paths(tmp_path):
    ckpt = tmp_path / "weird=path.pt"
    ckpt.write_bytes(b"weights")
    metrics = tmp_path / "m=etrics.json"
    metrics.write_text('{"NE": 120.0, "n": 20.0}\n')

    parsed = parse_candidate_json(
        json.dumps(
            {
                "step": 5000,
                "checkpoint": str(ckpt),
                "metrics_path": str(metrics),
                "sha256": _sha("eq"),
            }
        )
    )
    assert parsed["checkpoint"] == str(ckpt)
    assert parsed["metrics_path"] == str(metrics)


def test_parse_candidate_legacy_still_supported(tmp_path):
    ckpt = tmp_path / "a.pt"
    ckpt.write_bytes(b"x")
    metrics = tmp_path / "a.json"
    metrics.write_text('{"NE": 150.0, "n": 20.0}\n')

    parsed = parse_candidate_legacy(f"1000={ckpt}={metrics}={_sha('1')}")
    assert parsed["step"] == 1000
    assert parsed["mean_ne"] == 150.0


def test_cli_writes_baseline_lock_manifest(tmp_path):
    metrics_a = tmp_path / "a.json"
    metrics_a.write_text('{"NE": 150.0, "n": 20.0}\n')
    metrics_b = tmp_path / "b.json"
    metrics_b.write_text('{"NE": 120.0, "n": 20.0}\n')
    metrics_c = tmp_path / "c.json"
    metrics_c.write_text('{"NE": 120.0, "n": 20.0}\n')
    ckpt_a = tmp_path / "a.pt"
    ckpt_a.write_bytes(b"a")
    ckpt_b = tmp_path / "b.pt"
    ckpt_b.write_bytes(b"b")
    ckpt_c = tmp_path / "c.pt"
    ckpt_c.write_bytes(b"c")
    out = tmp_path / "baseline_lock.manifest.json"

    rc = main(
        [
            "--stamp",
            "test-stamp",
            "--candidate-json",
            json.dumps(
                {
                    "step": 1000,
                    "checkpoint": str(ckpt_a),
                    "metrics_path": str(metrics_a),
                    "sha256": _sha("1"),
                }
            ),
            "--candidate-json",
            json.dumps(
                {
                    "step": 4000,
                    "checkpoint": str(ckpt_b),
                    "metrics_path": str(metrics_b),
                    "sha256": _sha("2"),
                }
            ),
            "--candidate-json",
            json.dumps(
                {
                    "step": 5000,
                    "checkpoint": str(ckpt_c),
                    "metrics_path": str(metrics_c),
                    "sha256": _sha("3"),
                }
            ),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["checkpoint"] == str(ckpt_c)
    assert man["s1_ne"] == 96.0
    assert man["selection_rule"] == "min_mean_ne_tie_later_step"
    assert man["orchestration_version"] == ORCHESTRATION_VERSION
    datetime.fromisoformat(man["selection_time"])


def test_write_lock_manifest_atomic_roundtrip(tmp_path):
    out = tmp_path / "baseline_lock.manifest.json"
    payload = {"stamp": "x", "baseline_mean_ne": 1.0}
    write_lock_manifest(out, payload)
    assert read_status(out) == payload
    assert list(tmp_path.glob(".status.*")) == []
