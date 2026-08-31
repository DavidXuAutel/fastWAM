#!/usr/bin/env python3
"""Build ScoutXWAM 8-D proprio from FR3 EEF pose + gripper width."""
from __future__ import annotations

import numpy as np


def gripper_width_to_scout(finger_positions: np.ndarray) -> float:
    """Map Franka Hand finger joint widths (m) → Scout gripper scalar in [0, 1].

    Uses mean finger travel / 0.04 m (approx full open). Higher = more open.
    """
    w = np.asarray(finger_positions, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return 0.0
    mean_w = float(np.mean(w[:2])) if w.size >= 2 else float(w[0])
    return float(np.clip(mean_w / 0.04, 0.0, 1.0))


def build_scout_proprio(
    xyz: np.ndarray,
    quat_xyzw: np.ndarray,
    finger_positions: np.ndarray | float,
) -> np.ndarray:
    """Return float32 [8] = xyz(3) + quat_xyzw(4) + gripper(1)."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(3)
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    # Prefer w >= 0 for continuity
    if float(quat[3]) < 0:
        quat = -quat
    if np.isscalar(finger_positions):
        grip = float(np.clip(finger_positions, 0.0, 1.0))
    else:
        grip = gripper_width_to_scout(finger_positions)
    out = np.zeros(8, dtype=np.float32)
    out[0:3] = xyz
    out[3:7] = quat
    out[7] = grip
    return out


def scout_gripper_to_percent(g: float) -> float:
    """Scout gripper ~[0,1] open → Franka percent [0,1] with 1=open."""
    return float(np.clip(g, 0.0, 1.0))
