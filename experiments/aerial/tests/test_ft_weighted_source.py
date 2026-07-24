import logging
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from experiments.aerial.ft_mix_dataset import build_ft_mix_dataset
from experiments.aerial.ft_source_monitor import FTSourceMonitor
from fastwam.datasets.lerobot.weighted_source_dataset import WeightedSourceDataset
from fastwam.trainer import Wan22Trainer


class TinyDataset(Dataset):
    def __init__(self, name: str, length: int = 100):
        self.name = name
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"dataset": self.name, "index": index}


def test_weighted_source_dataset_draws_correction_at_25_percent():
    dataset = WeightedSourceDataset(
        datasets=[TinyDataset("original"), TinyDataset("correction")],
        probs=[0.75, 0.25],
        names=["original", "correction"],
        generator=torch.Generator().manual_seed(42),
    )

    for index in range(2_000):
        dataset[index]

    counts = dataset.pop_source_counts()
    correction_rate = counts["correction"] / sum(counts.values())
    assert 0.20 <= correction_rate <= 0.30
    assert dataset.pop_source_counts() == {"original": 0, "correction": 0}


def test_ft_seed_keeps_five_consecutive_windows_in_range():
    dataset = WeightedSourceDataset(
        datasets=[TinyDataset("original"), TinyDataset("correction")],
        probs=[0.75, 0.25],
        names=["original", "correction"],
        generator=torch.Generator().manual_seed(42),
    )

    correction_counts = []
    for index in range(1_000):
        dataset[index]
        if (index + 1) % 200 == 0:
            correction_counts.append(dataset.pop_source_counts()["correction"])

    assert correction_counts == [48, 57, 51, 55, 46]
    assert all(40 <= count <= 60 for count in correction_counts)


def test_ft_task_uses_main_process_dataset_rng():
    repo_root = Path(__file__).resolve().parents[3]
    task_cfg = OmegaConf.load(repo_root / "configs/task/aerial_joint_b0_ft_dagger.yaml")

    assert task_cfg.num_workers == 0


def test_weighted_source_dataset_samples_within_selected_source():
    dataset = WeightedSourceDataset(
        datasets=[TinyDataset("original", 3), TinyDataset("correction", 7)],
        probs=[0.0, 1.0],
        names=["original", "correction"],
        generator=torch.Generator().manual_seed(7),
    )

    samples = [dataset[index] for index in range(20)]

    assert len(dataset) == 10
    assert all(sample["dataset"] == "correction" for sample in samples)
    assert all(sample["data_source"] == "correction" for sample in samples)
    assert all(0 <= sample["index"] < 7 for sample in samples)


def test_ft_mix_target_builds_two_datasets(monkeypatch):
    built = []

    def fake_instantiate(config):
        built.append(config)
        return TinyDataset(config)

    monkeypatch.setattr("experiments.aerial.ft_mix_dataset.instantiate", fake_instantiate)

    dataset = build_ft_mix_dataset(
        original={"name": "original"},
        correction={"name": "correction"},
        source_probs=[0.75, 0.25],
        source_names=["original", "correction"],
        generator_seed=11,
    )

    assert built == [{"name": "original"}, {"name": "correction"}]
    assert isinstance(dataset, WeightedSourceDataset)


def test_ft_source_monitor_logs_counts_and_accepts_valid_window(caplog):
    monitor = FTSourceMonitor(
        correction_name="correction",
        log_every=50,
        window_steps=200,
        min_correction_rate=0.20,
        max_correction_rate=0.30,
    )

    with caplog.at_level(logging.INFO):
        for step in range(1, 201):
            source = "correction" if step % 4 == 0 else "original"
            monitor.record([source], step=step)

    assert "source counts steps=1-50" in caplog.text
    assert "correction_rate=0.2400" in caplog.text


@pytest.mark.parametrize("correction_steps", [39, 61])
def test_ft_source_monitor_rejects_out_of_range_window(correction_steps):
    monitor = FTSourceMonitor(
        correction_name="correction",
        log_every=50,
        window_steps=200,
        min_correction_rate=0.20,
        max_correction_rate=0.30,
    )

    with pytest.raises(RuntimeError, match="correction rate"):
        for step in range(1, 201):
            source = "correction" if step <= correction_steps else "original"
            monitor.record([source], step=step)


def test_ft_source_monitor_skips_partial_window_after_resume():
    monitor = FTSourceMonitor(
        correction_name="correction",
        log_every=50,
        window_steps=200,
        min_correction_rate=0.20,
        max_correction_rate=0.30,
    )
    monitor.reset(start_step=150)

    for step in range(151, 201):
        monitor.record(["original"], step=step)

    with pytest.raises(RuntimeError, match="steps 201-400"):
        for step in range(201, 401):
            source = "correction" if step < 240 else "original"
            monitor.record([source], step=step)


def test_trainer_records_collated_data_sources():
    calls = []

    class RecordingMonitor:
        def record(self, sources, *, step):
            calls.append((sources, step))

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.source_monitor = RecordingMonitor()
    trainer.global_step = 50

    trainer._record_ft_sources({"data_source": ["original", "correction"]})

    assert calls == [(["original", "correction"], 50)]


def test_trainer_resets_source_monitor_at_restored_step():
    calls = []

    class RecordingMonitor:
        def reset(self, *, start_step):
            calls.append(start_step)

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.source_monitor = RecordingMonitor()
    trainer.global_step = 150

    trainer._reset_source_monitor()

    assert calls == [150]
