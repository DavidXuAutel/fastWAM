#!/usr/bin/env python3
"""Adapters for running the released tau0-WM VAM interface in simulation.

The official tau0-WM server expects multi-view RGB observations, dual-arm EEF
poses, and gripper states. These helpers keep that contract explicit while
remaining independent from Isaac Sim so they can be unit-tested locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


def _as_float_vector(name: str, values: Sequence[float], length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _resize_nearest_hwc(rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Dependency-light nearest-neighbor resize for uint8/float HWC images."""
    out_h, out_w = size_hw
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {rgb.shape}")
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid output size: {size_hw}")
    src_h, src_w = rgb.shape[:2]
    y_idx = np.linspace(0, src_h - 1, out_h).round().astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, out_w).round().astype(np.int64)
    return rgb[y_idx][:, x_idx]


def quat_xyzw_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = _as_float_vector("quat_xyzw", quat_xyzw, 4).astype(np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("quat_xyzw must not be zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _normalize_gripper(open_fraction: float, *, max_value: float = 120.0) -> float:
    if not np.isfinite(open_fraction):
        raise ValueError("gripper fraction must be finite")
    clipped = float(np.clip(open_fraction, 0.0, 1.0))
    return clipped * max_value


@dataclass(frozen=True)
class Tau0ObservationAdapter:
    """Convert simulator camera frames to tau0-WM `obs` payload."""

    image_size: Tuple[int, int] = (192, 256)  # H, W

    def to_payload_obs(self, views: Iterable[np.ndarray]) -> np.ndarray:
        processed = []
        for view in views:
            arr = np.asarray(view)
            if arr.ndim != 3:
                raise ValueError(f"each view must be RGB image, got shape {arr.shape}")
            if arr.shape[0] == 3 and arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] != 3:
                raise ValueError(f"each view must have 3 channels, got shape {arr.shape}")

            resized = _resize_nearest_hwc(arr, self.image_size)
            if resized.dtype == np.uint8:
                norm = resized.astype(np.float32) / 127.5 - 1.0
            else:
                norm = resized.astype(np.float32)
                if norm.min(initial=0.0) >= 0.0 and norm.max(initial=0.0) > 1.0:
                    norm = norm / 127.5 - 1.0
                elif norm.min(initial=0.0) >= 0.0 and norm.max(initial=0.0) <= 1.0:
                    norm = norm * 2.0 - 1.0
            processed.append(np.transpose(np.clip(norm, -1.0, 1.0), (2, 0, 1)))
        if not processed:
            raise ValueError("at least one camera view is required")
        return np.stack(processed, axis=0).astype(np.float32)


@dataclass(frozen=True)
class Tau0StateAdapter:
    """Build the released tau0-WM dual-EEF state contract."""

    gripper_max_value: float = 120.0

    def to_tau0_state(
        self,
        *,
        left_position: Sequence[float],
        left_quat_xyzw: Sequence[float],
        right_position: Sequence[float],
        right_quat_xyzw: Sequence[float],
        left_gripper_open: float,
        right_gripper_open: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_pos = _as_float_vector("left_position", left_position, 3)
        left_quat = _as_float_vector("left_quat_xyzw", left_quat_xyzw, 4)
        right_pos = _as_float_vector("right_position", right_position, 3)
        right_quat = _as_float_vector("right_quat_xyzw", right_quat_xyzw, 4)
        state = np.concatenate([left_pos, left_quat, right_pos, right_quat]).astype(np.float32)
        grippers = np.asarray(
            [
                _normalize_gripper(left_gripper_open, max_value=self.gripper_max_value),
                _normalize_gripper(right_gripper_open, max_value=self.gripper_max_value),
            ],
            dtype=np.float32,
        )
        return state, grippers


@dataclass(frozen=True)
class Tau0ExecutionChunk:
    left_position: np.ndarray
    left_quat_xyzw: np.ndarray
    left_gripper: np.ndarray
    right_position: np.ndarray
    right_quat_xyzw: np.ndarray
    right_gripper: np.ndarray
    raw: np.ndarray


@dataclass(frozen=True)
class Tau0ActionAdapter:
    """Validate and slice tau0-WM `[T,16]` action chunks for execution."""

    execution_steps: int = 10
    gripper_max_value: float = 120.0

    def prepare_execution_chunk(self, actions: np.ndarray) -> Tau0ExecutionChunk:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 16:
            raise ValueError(f"tau0 actions must have shape [T,16], got {arr.shape}")
        if self.execution_steps <= 0:
            raise ValueError("execution_steps must be positive")
        if not np.all(np.isfinite(arr)):
            raise ValueError("actions contain non-finite values")
        chunk = arr[: min(self.execution_steps, arr.shape[0])].copy()
        return Tau0ExecutionChunk(
            left_position=chunk[:, 0:3],
            left_quat_xyzw=chunk[:, 3:7],
            left_gripper=np.clip(chunk[:, 7] / self.gripper_max_value, 0.0, 1.0),
            right_position=chunk[:, 8:11],
            right_quat_xyzw=chunk[:, 11:15],
            right_gripper=np.clip(chunk[:, 15] / self.gripper_max_value, 0.0, 1.0),
            raw=chunk,
        )
