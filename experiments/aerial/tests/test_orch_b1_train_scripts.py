from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "experiments" / "aerial" / "scripts"


def _run(script: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        check=False,
    )


def test_accelerate_h100_config_is_zero2_no_offload() -> None:
    config = (SCRIPTS / "accelerate_zero2_no_offload_2proc.yaml").read_text(encoding="utf-8")
    assert "distributed_type: DEEPSPEED" in config
    assert "zero_stage: 2" in config
    assert "offload_optimizer_device: none" in config
    assert "num_processes: 2" in config
    assert "mixed_precision: bf16" in config


def test_orch_b1_train_dry_run_locks_recipe(tmp_path: Path) -> None:
    cache = tmp_path / "ft_cache"
    repo = cache / "repo"
    scripts = repo / "experiments" / "aerial" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "accelerate_zero2_no_offload_2proc.yaml").write_text(
        (SCRIPTS / "accelerate_zero2_no_offload_2proc.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    model = cache / "model"
    model.mkdir()
    ckpt = model / "baseline.pt"
    ckpt.write_bytes(b"weights")
    orch = tmp_path / "orch"
    orch.mkdir()
    status = orch / "status.json"
    status.write_text(
        '{"phase":"RUN_B1_TRAIN","stamp":"s1","gates_passed":true}\n',
        encoding="utf-8",
    )
    lock = orch / "baseline_lock.manifest.json"
    lock.write_text(
        '{"checkpoint":"%s","sha256":"%s"}\n'
        % (ckpt, "0" * 64),
        encoding="utf-8",
    )
    (cache / "smoke.status").write_text("PASSED\n", encoding="utf-8")
    (cache / "SHA256SUMS").write_text("deadbeef  model/baseline.pt\n", encoding="utf-8")

    result = _run(
        "orch_b1_train.sh",
        "--dry-run",
        env={
            "AERIAL_FT_CACHE": str(cache),
            "STATUS_PATH": str(status),
            "LOCK_PATH": str(lock),
            "SKIP_MANIFEST_VERIFY": "1",
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    out = result.stdout
    assert "lambda_video=0.0" in out
    assert "max_steps=1000" in out
    assert "save_every=250" in out
    assert "learning_rate=1e-5" in out
    assert "resume=" in out and "baseline.pt" in out


def test_orch_b1_train_refuses_without_gates(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text(
        '{"phase":"B1_GATES","stamp":"s1","gates_passed":false}\n',
        encoding="utf-8",
    )
    result = _run(
        "orch_b1_train.sh",
        "--dry-run",
        env={
            "AERIAL_FT_CACHE": str(tmp_path / "missing"),
            "STATUS_PATH": str(status),
            "LOCK_PATH": str(tmp_path / "missing_lock.json"),
        },
    )
    assert result.returncode != 0
    assert "gates" in (result.stderr + result.stdout).lower()


def test_sync_h100_dry_run_targets_31103(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "step_002000.pt"
    checkpoint.write_bytes(b"weights")
    (checkpoint_dir / "dataset_stats.json").write_text("{}", encoding="utf-8")
    assets = {}
    for name in ("train_subset", "correction", "text_embeds"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "asset.bin").write_bytes(name.encode())
        assets[name] = directory
    collection_manifest = tmp_path / "collection_manifest.json"
    collection_manifest.write_text("{}", encoding="utf-8")
    lock = tmp_path / "baseline_lock.manifest.json"
    lock.write_text(
        '{"checkpoint":"%s","sha256":"%s"}\n' % (checkpoint, "a" * 64),
        encoding="utf-8",
    )

    result = _run(
        "sync_b0_ft_to_h100.sh",
        "--dry-run",
        env={
            "LOCK_PATH": str(lock),
            "TRAIN_SUBSET": str(assets["train_subset"]),
            "CORRECTION_SET": str(assets["correction"]),
            "TEXT_EMBEDS": str(assets["text_embeds"]),
            "COLLECTION_MANIFEST": str(collection_manifest),
            "CODE_ROOT": str(REPO_ROOT),
            "DATASET_STATS": str(checkpoint_dir / "dataset_stats.json"),
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY RUN: no SSH or rsync commands executed" in result.stdout
    assert "a25689@10.239.121.22" in result.stdout
    assert "SSH port 31103" in result.stdout


def test_ckpt_watch_dry_run_lists_b1_steps() -> None:
    result = _run("orch_ckpt_watch_enqueue.sh", "--dry-run", env={})
    assert result.returncode == 0, result.stderr + result.stdout
    for step in (250, 500, 1000):
        assert f"step_{step:06d}" in result.stdout


def test_shell_scripts_parse() -> None:
    for name in (
        "orch_b1_train.sh",
        "orch_ckpt_watch_enqueue.sh",
        "sync_b0_ft_to_h100.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
