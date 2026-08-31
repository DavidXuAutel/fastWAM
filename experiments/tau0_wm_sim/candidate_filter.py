#!/usr/bin/env python3
"""Lightweight action-candidate filtering for tau0-WM sim-first deployment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateFilterConfig:
    max_eef_step_m: float = 0.20
    max_rotation_step_norm: float = 1.0
    min_gripper_value: float = 0.0
    max_gripper_value: float = 120.0
    velocity_penalty: float = 1.0
    rotation_penalty: float = 0.25
    violation_penalty: float = 100.0
    gripper_violation_penalty: float = 1.0


@dataclass(frozen=True)
class CandidateScore:
    index: int
    score: float
    valid: bool
    max_eef_step_m: float
    max_rotation_step_norm: float
    violations: Sequence[str]


def _validate_action(candidate: np.ndarray) -> np.ndarray:
    arr = np.asarray(candidate, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"candidate must have shape [T,16], got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("candidate must contain at least 2 timesteps")
    if not np.all(np.isfinite(arr)):
        raise ValueError("candidate contains non-finite values")
    return arr


def score_action_candidate(
    candidate: np.ndarray,
    config: CandidateFilterConfig = CandidateFilterConfig(),
    *,
    index: int = 0,
) -> CandidateScore:
    arr = _validate_action(candidate)
    left_delta = np.diff(arr[:, 0:3], axis=0)
    right_delta = np.diff(arr[:, 8:11], axis=0)
    left_step = np.linalg.norm(left_delta, axis=1)
    right_step = np.linalg.norm(right_delta, axis=1)
    max_eef_step = float(max(left_step.max(initial=0.0), right_step.max(initial=0.0)))

    left_rot = np.linalg.norm(np.diff(arr[:, 3:7], axis=0), axis=1)
    right_rot = np.linalg.norm(np.diff(arr[:, 11:15], axis=0), axis=1)
    max_rot_step = float(max(left_rot.max(initial=0.0), right_rot.max(initial=0.0)))

    violations: List[str] = []
    if max_eef_step > config.max_eef_step_m:
        violations.append("eef_step")
    if max_rot_step > config.max_rotation_step_norm:
        violations.append("rotation_step")
    grippers = arr[:, [7, 15]]
    if grippers.min(initial=config.min_gripper_value) < config.min_gripper_value or grippers.max(
        initial=config.max_gripper_value
    ) > config.max_gripper_value:
        violations.append("gripper_range")

    smooth_cost = config.velocity_penalty * max_eef_step + config.rotation_penalty * max_rot_step
    violation_cost = config.violation_penalty * len(violations)
    if "gripper_range" in violations:
        violation_cost += config.gripper_violation_penalty
    score = -smooth_cost - violation_cost
    return CandidateScore(
        index=index,
        score=float(score),
        valid=not violations,
        max_eef_step_m=max_eef_step,
        max_rotation_step_norm=max_rot_step,
        violations=tuple(violations),
    )


def rank_action_candidates(
    candidates: Iterable[np.ndarray],
    config: CandidateFilterConfig = CandidateFilterConfig(),
) -> List[CandidateScore]:
    scored = [
        score_action_candidate(candidate, config, index=i)
        for i, candidate in enumerate(candidates)
    ]
    return sorted(scored, key=lambda item: item.score, reverse=True)
