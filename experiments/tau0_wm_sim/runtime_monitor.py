#!/usr/bin/env python3
"""Runtime measurement utilities for tau0-WM deployment experiments."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass(frozen=True)
class RuntimeSample:
    name: str
    latency_s: float
    gpu_memory_mb: Optional[float] = None


class RuntimeRecorder:
    def __init__(self) -> None:
        self._samples: List[RuntimeSample] = []

    def add_sample(
        self,
        *,
        name: str,
        latency_s: float,
        gpu_memory_mb: Optional[float] = None,
    ) -> None:
        if latency_s < 0:
            raise ValueError("latency_s must be non-negative")
        self._samples.append(RuntimeSample(name=name, latency_s=float(latency_s), gpu_memory_mb=gpu_memory_mb))

    @contextmanager
    def measure(self, name: str, *, gpu_memory_mb: Optional[float] = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_sample(name=name, latency_s=time.perf_counter() - start, gpu_memory_mb=gpu_memory_mb)

    def summary(self) -> Dict[str, object]:
        latencies = [sample.latency_s for sample in self._samples]
        memory = [sample.gpu_memory_mb for sample in self._samples if sample.gpu_memory_mb is not None]
        return {
            "samples": len(self._samples),
            "latency_s": _stats(latencies),
            "gpu_memory_mb": _stats(memory),
            "by_name": _by_name(self._samples),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _by_name(samples: List[RuntimeSample]) -> Dict[str, Dict[str, Optional[float]]]:
    grouped: Dict[str, List[float]] = {}
    for sample in samples:
        grouped.setdefault(sample.name, []).append(sample.latency_s)
    return {name: _stats(values) for name, values in grouped.items()}
