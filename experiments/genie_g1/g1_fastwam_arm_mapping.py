#!/usr/bin/env python3
"""
G1 embodiment ↔ FastWAM arm layout mapping (standalone, no ROS import required).

Context (from this repo)
-------------------------
1) **G1 HAL / WBC (Genie GDK)** — same ordering as ``GenieG1TaskEnv`` in
   ``experiments/genie_g1/ros_g1_bridge.py``:
   - ``/hal/arm_joint_state``: 14 floats, **radians**, **left arm joints 0–6**
     then **right arm joints 7–13**.
   - ``/wbc/arm_command``: ``sensor_msgs/JointState.position`` with the **same**
     14 order when commanding absolute joint targets.

2) **FastWAM infer observation** — ``fastwam-g01-bridge`` builds JSON via
   ``snapshot_to_observation_json`` → ``floats_to_observation_fields``:
   - ``left_state``: length-7 list (radians)
   - ``right_state``: length-7 list (radians)
   i.e. split of G1 ``arm14`` as ``arm14[0:7]`` and ``arm14[7:14]``.

3) **FastWAM action** — ``expand_action_to_arm14`` in
   ``fastwam-g01-bridge/fastwam_g01_bridge/action_mapper.py``:
   - dim 14: passthrough to G1 command vector
   - dim 7: duplicate / left_only / right_only modes to fill 14 targets

This module implements those mappings explicitly so tools/tests can convert
without importing the bridge package.

Notes
-----
- If ``JointState.name`` is present on the robot, joint *semantic* order should
  match HAL’s publication order; index ``i`` always refers to that order.
- For a different physical permutation (URDF vs HAL), apply a fixed index
  permutation **after** ``arm14`` conversion (see ``permute_arm14``).

Usage
-----
  python3 g1_fastwam_arm_mapping.py selftest
  python3 g1_fastwam_arm_mapping.py show-layout
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Sequence, Tuple

# --- Layout constants (G1 dual 7-DoF arms, radians) ---

G1_ARM_DIM = 14
G1_LEFT_INDICES = tuple(range(0, 7))
G1_RIGHT_INDICES = tuple(range(7, 14))


def _as_float_list(x: Sequence[float], n: int, label: str) -> List[float]:
    if len(x) != n:
        raise ValueError(f"{label}: expected length {n}, got {len(x)}")
    return [float(v) for v in x]


def g1_hal_arm14_to_fastwam_observation(
    arm14: Sequence[float],
) -> Dict[str, List[float]]:
    """Map G1 HAL ``arm14`` to FastWAM observation-style ``left_state`` / ``right_state``."""
    v = _as_float_list(arm14, G1_ARM_DIM, "arm14")
    return {"left_state": v[0:7], "right_state": v[7:14]}


def fastwam_observation_to_g1_hal_arm14(obs: Dict[str, Any]) -> List[float]:
    """Invert ``g1_hal_arm14_to_fastwam_observation`` for plain list obs."""
    if "left_state" not in obs or "right_state" not in obs:
        raise KeyError("obs must contain 'left_state' and 'right_state'")
    left = _as_float_list(obs["left_state"], 7, "left_state")
    right = _as_float_list(obs["right_state"], 7, "right_state")
    return left + right


def split_arm14(arm14: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Return (left7, right7) from G1 ``arm14``."""
    v = _as_float_list(arm14, G1_ARM_DIM, "arm14")
    return v[0:7], v[7:14]


def merge_arm14(left7: Sequence[float], right7: Sequence[float]) -> List[float]:
    """Merge FastWAM-style halves to G1 ``arm14``."""
    l = _as_float_list(left7, 7, "left7")
    r = _as_float_list(right7, 7, "right7")
    return l + r


def expand_fastwam_action_to_g1_hal_arm14(
    action: Sequence[float],
    *,
    mode: str,
    current_arm14: Sequence[float],
) -> List[float]:
    """
    Mirror ``fastwam_g01_bridge.action_mapper.expand_action_to_arm14``.

    Maps policy output (7 or 14 floats, radians) → G1 command ``arm14``.
    """
    act = [float(x) for x in action]
    n = len(act)
    cur = _as_float_list(current_arm14, G1_ARM_DIM, "current_arm14")

    if n == G1_ARM_DIM:
        return act
    if n != 7:
        raise ValueError(f"FastWAM action dim {n} not supported (expected 7 or 14)")

    left = act[:7]
    right = act[:7]
    m = mode.lower()
    if m == "duplicate":
        return left + right
    if m == "left_only":
        return left + cur[7:14]
    if m == "right_only":
        return cur[0:7] + right
    raise ValueError(f"unknown ACTION_7DOF_MODE: {mode!r}")


def permute_arm14(arm14: Sequence[float], indices: Sequence[int]) -> List[float]:
    """
    Re-order a 14-vector by ``indices[i] = source index for output slot i``.

    Use when HAL order differs from another convention (advanced).
    """
    v = _as_float_list(arm14, G1_ARM_DIM, "arm14")
    if len(indices) != G1_ARM_DIM:
        raise ValueError("indices must have length 14")
    out = [0.0] * G1_ARM_DIM
    for out_i, src_i in enumerate(indices):
        if src_i < 0 or src_i >= G1_ARM_DIM:
            raise ValueError(f"bad index {src_i}")
        out[out_i] = v[src_i]
    return out


def layout_reference_text() -> str:
    return (
        "G1 HAL arm14 index layout (radians, left then right):\n"
        "  [0:7]   left arm  -> FastWAM observation key 'left_state'\n"
        "  [7:14]  right arm -> FastWAM observation key 'right_state'\n"
        "Matches /hal/arm_joint_state and /wbc/arm_command position order "
        "(see ros_g1_bridge.py).\n"
    )


def _selftest() -> int:
    arm = [float(i) * 0.1 for i in range(14)]
    obs = g1_hal_arm14_to_fastwam_observation(arm)
    back = fastwam_observation_to_g1_hal_arm14(obs)
    assert back == arm, (back, arm)

    cur = [1.0] * 14
    assert expand_fastwam_action_to_g1_hal_arm14([2.0] * 7, mode="duplicate", current_arm14=cur) == [2.0] * 14
    assert expand_fastwam_action_to_g1_hal_arm14([3.0] * 7, mode="left_only", current_arm14=cur)[:7] == [3.0] * 7
    assert expand_fastwam_action_to_g1_hal_arm14([4.0] * 7, mode="left_only", current_arm14=cur)[7:] == [1.0] * 7
    assert expand_fastwam_action_to_g1_hal_arm14([5.0] * 7, mode="right_only", current_arm14=cur)[:7] == [1.0] * 7
    assert expand_fastwam_action_to_g1_hal_arm14([6.0] * 7, mode="right_only", current_arm14=cur)[7:] == [6.0] * 7

    ident = list(range(14))
    assert permute_arm14(ident, list(range(14))) == ident
    print("selftest: OK")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G1 ↔ FastWAM arm vector mapping")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="run consistency checks")

    p_show = sub.add_parser("show-layout", help="print layout reference")

    p_obs = sub.add_parser("hal-to-obs", help="arm14 JSON -> left_state/right_state JSON")
    p_obs.add_argument("--json", required=True, help='JSON object with key "arm14": list of 14 floats')

    p_hal = sub.add_parser("obs-to-hal", help="observation JSON -> arm14 JSON")
    p_hal.add_argument(
        "--json",
        required=True,
        help='JSON with "left_state" (7) and "right_state" (7)',
    )

    p_act = sub.add_parser(
        "action-to-hal",
        help="FastWAM action (7 or 14) -> arm14 using ACTION_7DOF_MODE",
    )
    p_act.add_argument("--json", required=True, help='JSON: {"action": [...], "current_arm14": [...]}')
    p_act.add_argument(
        "--mode",
        default="duplicate",
        choices=("duplicate", "left_only", "right_only"),
    )

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return _selftest()
    if args.cmd == "show-layout":
        print(layout_reference_text())
        return 0
    if args.cmd == "hal-to-obs":
        data = json.loads(args.json)
        arm14 = data["arm14"]
        out = g1_hal_arm14_to_fastwam_observation(arm14)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "obs-to-hal":
        data = json.loads(args.json)
        arm14 = fastwam_observation_to_g1_hal_arm14(data)
        json.dump({"arm14": arm14}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.cmd == "action-to-hal":
        data = json.loads(args.json)
        arm14 = expand_fastwam_action_to_g1_hal_arm14(
            data["action"],
            mode=args.mode,
            current_arm14=data["current_arm14"],
        )
        json.dump({"arm14": arm14}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
