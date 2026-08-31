#!/usr/bin/env python3
"""HTTP client for ScoutXWAM Franka infer service."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import requests


@dataclass
class ScoutClient:
    base_url: str = "http://127.0.0.1:8010"
    timeout_s: float = 600.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=min(30.0, self.timeout_s))
        resp.raise_for_status()
        return resp.json()

    def infer(
        self,
        *,
        video: np.ndarray,
        proprio: np.ndarray,
        prompt: str,
        seed: int = 42,
        action_denoise_steps: int = 10,
    ) -> dict[str, Any]:
        video = np.asarray(video, dtype=np.uint8)
        if video.ndim != 4 or video.shape[0] != 2 or video.shape[-1] != 3:
            raise ValueError(f"video must be [2,H,W,3] uint8, got {video.shape}/{video.dtype}")
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if proprio.size not in (8, 16):
            raise ValueError(f"proprio must be 8 or 16, got {proprio.size}")
        payload = {
            "prompt": prompt,
            "proprio": proprio.astype(float).tolist(),
            "video_shape": list(video.shape),
            "video_b64": base64.b64encode(np.ascontiguousarray(video).tobytes()).decode("ascii"),
            "seed": int(seed),
            "action_denoise_steps": int(action_denoise_steps),
        }
        resp = requests.post(
            f"{self.base_url}/v1/infer",
            json=payload,
            timeout=self.timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"infer failed {resp.status_code}: {resp.text[:500]}")
        return resp.json()


def default_base_url() -> str:
    return os.environ.get("SCOUT_URL", "http://127.0.0.1:8010")
