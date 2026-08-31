#!/usr/bin/env python3
"""VAM-only simulation loop scaffold for tau0-WM.

This script is intentionally Isaac-light: it defines the policy-side loop and
payload contract. Replace `MockSimBridge` with an Isaac bridge that returns RGB
views, EEF poses, and gripper fractions from the live stage.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from adapters import Tau0ActionAdapter, Tau0ObservationAdapter, Tau0StateAdapter
from runtime_monitor import RuntimeRecorder


class Tau0PolicyClient(Protocol):
    def infer(self, *, obs: dict) -> dict:
        ...


@dataclass
class MockSimState:
    left_position: np.ndarray
    left_quat_xyzw: np.ndarray
    right_position: np.ndarray
    right_quat_xyzw: np.ndarray
    left_gripper_open: float
    right_gripper_open: float


class MockSimBridge:
    """A deterministic bridge used for local smoke tests before Isaac wiring."""

    def __init__(self, *, views: int = 3, image_size_hw: tuple[int, int] = (192, 256)) -> None:
        self.views = views
        self.image_size_hw = image_size_hw
        self.step_count = 0

    def read_rgb_views(self) -> Sequence[np.ndarray]:
        h, w = self.image_size_hw
        frames = []
        for i in range(self.views):
            value = np.uint8((self.step_count * 13 + i * 50) % 255)
            frames.append(np.full((h, w, 3), value, dtype=np.uint8))
        return frames

    def read_state(self) -> MockSimState:
        offset = self.step_count * 0.001
        return MockSimState(
            left_position=np.asarray([0.3 + offset, 0.2, 0.4], dtype=np.float32),
            left_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            right_position=np.asarray([0.3 + offset, -0.2, 0.4], dtype=np.float32),
            right_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            left_gripper_open=1.0,
            right_gripper_open=1.0,
        )

    def apply_action_chunk(self, actions: np.ndarray) -> None:
        self.step_count += max(1, int(actions.shape[0]))


class MockPolicyClient:
    def infer(self, *, obs: dict) -> dict:
        horizon = int(obs.get("execution_step", 10))
        actions = np.zeros((horizon, 16), dtype=np.float32)
        actions[:, 3] = 0.0
        actions[:, 6] = 1.0
        actions[:, 11] = 0.0
        actions[:, 14] = 1.0
        actions[:, 7] = 120.0
        actions[:, 15] = 120.0
        return {"actions": actions}


class OfficialWebsocketPolicyClient:
    def __init__(self, *, tau_repo: Path, host: str, port: int) -> None:
        sys.path.insert(0, str(tau_repo))
        from web_infer_utils.openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def infer(self, *, obs: dict) -> dict:
        return self._client.infer(obs=obs)


def run_vam_only_loop(
    *,
    policy: Tau0PolicyClient,
    sim: MockSimBridge,
    prompt: str,
    iterations: int,
    execution_step: int,
    output_report: Optional[Path] = None,
) -> dict:
    obs_adapter = Tau0ObservationAdapter()
    state_adapter = Tau0StateAdapter()
    action_adapter = Tau0ActionAdapter(execution_steps=execution_step)
    recorder = RuntimeRecorder()

    last_chunk = None
    for _ in range(iterations):
        sim_state = sim.read_state()
        state, grippers = state_adapter.to_tau0_state(
            left_position=sim_state.left_position,
            left_quat_xyzw=sim_state.left_quat_xyzw,
            right_position=sim_state.right_position,
            right_quat_xyzw=sim_state.right_quat_xyzw,
            left_gripper_open=sim_state.left_gripper_open,
            right_gripper_open=sim_state.right_gripper_open,
        )
        payload = {
            "obs": obs_adapter.to_payload_obs(sim.read_rgb_views()),
            "prompt": prompt,
            "state": state,
            "gripper_states": grippers,
            "num_inference_steps": 5,
            "execution_step": execution_step,
            "sample_solver": "euler",
            "shift": 1.0,
        }
        with recorder.measure("policy_infer"):
            result = policy.infer(obs=payload)
        raw_actions = np.asarray(result["actions"], dtype=np.float32)
        chunk = action_adapter.prepare_execution_chunk(raw_actions)
        last_chunk = chunk.raw
        sim.apply_action_chunk(chunk.raw)

    report = recorder.summary()
    report["iterations"] = iterations
    report["last_action_shape"] = list(last_chunk.shape) if last_chunk is not None else None
    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        import json

        output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--execution-step", type=int, default=10)
    parser.add_argument("--mock", action="store_true", help="Run without a tau0-WM server.")
    parser.add_argument("--tau-repo", type=Path, help="Path to cloned sii-research/tau-0-wm repo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    if args.mock:
        policy: Tau0PolicyClient = MockPolicyClient()
    else:
        if args.tau_repo is None:
            raise SystemExit("--tau-repo is required unless --mock is set")
        policy = OfficialWebsocketPolicyClient(tau_repo=args.tau_repo, host=args.host, port=args.port)

    report = run_vam_only_loop(
        policy=policy,
        sim=MockSimBridge(),
        prompt=args.prompt,
        iterations=args.iterations,
        execution_step=args.execution_step,
        output_report=args.output_report,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
