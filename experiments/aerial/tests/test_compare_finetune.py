import json
from pathlib import Path
import subprocess

from experiments.aerial.eval.compare_finetune import (
    S1_NE_THRESHOLD,
    compare_metrics,
    main,
    summarize,
)


def test_s1_pass_and_fail():
    report = summarize(
        baseline_ne=135.94562291546043,
        cand={"250": 120.0, "500": 100.0},
    )
    assert report["best_step"] == "500"
    assert report["s1_pass"] is True

    failed = summarize(
        baseline_ne=135.94562291546043,
        cand={"250": S1_NE_THRESHOLD + 0.01},
    )
    assert failed["s1_pass"] is False
    assert failed["diagnosis"]["failure_bins"] == [
        "improved_but_below_s1_margin",
        "flat",
        "regressed",
        "quantization_gap",
    ]
    boundary = summarize(
        baseline_ne=135.94562291546043,
        cand={"1000": S1_NE_THRESHOLD},
    )
    assert boundary["s1_pass"] is True


def test_compare_metrics_includes_episode_deltas_and_quantization_stats():
    baseline = {
        "NE": 30.0,
        "SR": 0.25,
        "SPL": 0.20,
        "episodes": [
            {"episode_id": "route-a", "NE": 10.0},
            {"episode_id": "route-b", "NE": 30.0},
            {"episode_id": "route-c", "NE": 50.0},
        ],
    }
    candidate = {
        "NE": 25.0,
        "SR": 0.50,
        "SPL": 0.40,
        "episodes": [
            {"episode_id": "route-a", "NE": 8.0},
            {"episode_id": "route-b", "NE": 30.0},
            {"episode_id": "route-c", "NE": 55.0},
        ],
        "quantization_gap_l2": [1.0, 2.0, 3.0],
    }

    report = compare_metrics(baseline, {"250": candidate})
    run = report["candidates"]["250"]
    assert run["mean_NE"] == 25.0
    assert run["median_NE"] == 30.0
    assert run["SR"] == 0.50
    assert run["SPL"] == 0.40
    assert run["per_episode_deltas"] == {
        "route-a": -2.0,
        "route-b": 0.0,
        "route-c": 5.0,
    }
    assert run["delta_counts"] == {"improve": 1, "flat": 1, "regress": 1}
    assert run["quantization_gap_l2"] == {
        "count": 3,
        "mean": 2.0,
        "median": 2.0,
        "max": 3.0,
    }


def test_cli_fail_writes_report_and_diagnosis_scaffold(tmp_path: Path):
    baseline = tmp_path / "step_004000.json"
    candidate = tmp_path / "step_000250.json"
    out = tmp_path / "ft_selection_report.json"
    diagnosis = tmp_path / "ft_s1_failure_diagnosis.json"
    baseline.write_text(json.dumps({"NE": 135.94562291546043}), encoding="utf-8")
    candidate.write_text(json.dumps({"NE": 120.0}), encoding="utf-8")

    rc = main(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            f"250={candidate}",
            "--out",
            str(out),
            "--diagnosis-out",
            str(diagnosis),
        ]
    )

    assert rc == 1
    assert json.loads(out.read_text(encoding="utf-8"))["s1_pass"] is False
    payload = json.loads(diagnosis.read_text(encoding="utf-8"))
    assert payload["auto_expand_data"] is False
    assert payload["start_unseen"] is False


def test_cli_rejects_baseline_ne_that_is_not_locked(tmp_path: Path):
    baseline = tmp_path / "step_004000.json"
    candidate = tmp_path / "step_000250.json"
    baseline.write_text(json.dumps({"NE": 135.0}), encoding="utf-8")
    candidate.write_text(json.dumps({"NE": 100.0}), encoding="utf-8")

    try:
        main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                f"250={candidate}",
                "--out",
                str(tmp_path / "report.json"),
            ]
        )
    except ValueError as exc:
        assert "locked baseline" in str(exc)
    else:
        raise AssertionError("unlocked baseline NE was accepted")


def test_eval_script_dry_run_locks_heldout_protocol(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_ft_ckpts_seen20.sh"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_DIR": str(tmp_path / "repo"),
            "OPENFLY_ROOT": str(tmp_path / "openfly"),
            "HELDOUT_ANN": str(tmp_path / "heldout_seen20.json"),
            "B0_METRICS": str(tmp_path / "step_004000.json"),
            "FT_RUN_DIR": str(tmp_path / "ft"),
            "RESULT_DIR": str(tmp_path / "results"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("run_closed_loop") == 3
    for step in (250, 500, 1000):
        assert f"step_{step:06d}.pt" in result.stdout
    assert "--max-episodes 20" in result.stdout
    assert "--max-steps 100" in result.stdout
    assert "--seed 42" in result.stdout
    assert "--task aerial_joint_b0_novideo" in result.stdout
    assert "compare_finetune" in result.stdout
    assert "unseen" not in result.stdout.lower()
