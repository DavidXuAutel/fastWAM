from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import (  # noqa: E402
    Tau0ActionAdapter,
    Tau0ObservationAdapter,
    Tau0StateAdapter,
    quat_xyzw_to_matrix,
)
from acvs_status import inspect_acvs_readiness  # noqa: E402
from candidate_filter import CandidateFilterConfig, rank_action_candidates  # noqa: E402
from preflight import run_preflight  # noqa: E402
from runtime_monitor import RuntimeRecorder  # noqa: E402


def test_observation_adapter_normalizes_uint8_hwc_views_to_vchw() -> None:
    view0 = np.zeros((4, 8, 3), dtype=np.uint8)
    view1 = np.full((4, 8, 3), 255, dtype=np.uint8)

    payload = Tau0ObservationAdapter(image_size=(2, 4)).to_payload_obs([view0, view1])

    assert payload.shape == (2, 3, 2, 4)
    assert payload.dtype == np.float32
    assert np.isclose(payload[0].min(), -1.0)
    assert np.isclose(payload[1].max(), 1.0)


def test_state_adapter_builds_official_14d_state_and_gripper_range() -> None:
    state_adapter = Tau0StateAdapter()

    state, gripper = state_adapter.to_tau0_state(
        left_position=[0.1, 0.2, 0.3],
        left_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        right_position=[0.4, 0.5, 0.6],
        right_quat_xyzw=[0.0, 0.0, 1.0, 0.0],
        left_gripper_open=0.25,
        right_gripper_open=1.0,
    )

    assert state.shape == (14,)
    assert gripper.tolist() == [30.0, 120.0]
    assert np.allclose(state[:3], [0.1, 0.2, 0.3])
    assert np.allclose(state[7:10], [0.4, 0.5, 0.6])


def test_quat_xyzw_to_matrix_normalizes_input() -> None:
    matrix = quat_xyzw_to_matrix([0.0, 0.0, 0.0, 2.0])

    assert np.allclose(matrix, np.eye(3), atol=1e-6)


def test_action_adapter_slices_chunk_and_maps_gripper_to_unit_interval() -> None:
    raw = np.zeros((5, 16), dtype=np.float32)
    raw[:, 0] = np.linspace(0.0, 0.4, 5)
    raw[:, 7] = np.linspace(0.0, 120.0, 5)
    raw[:, 15] = 60.0

    chunk = Tau0ActionAdapter(execution_steps=3).prepare_execution_chunk(raw)

    assert chunk.left_position.shape == (3, 3)
    assert np.allclose(chunk.left_gripper, [0.0, 0.25, 0.5])
    assert np.allclose(chunk.right_gripper, [0.5, 0.5, 0.5])


def test_candidate_filter_prefers_smooth_valid_candidate() -> None:
    base = np.zeros((4, 16), dtype=np.float32)
    smooth = base.copy()
    smooth[:, 0] = [0.0, 0.01, 0.02, 0.03]
    jumpy = base.copy()
    jumpy[:, 0] = [0.0, 0.4, 0.8, 1.2]
    invalid_gripper = smooth.copy()
    invalid_gripper[:, 7] = 200.0

    ranked = rank_action_candidates(
        [jumpy, smooth, invalid_gripper],
        CandidateFilterConfig(max_eef_step_m=0.2, max_gripper_value=120.0),
    )

    assert ranked[0].index == 1
    assert ranked[0].valid
    assert not ranked[-1].valid
    assert "gripper_range" in ranked[-1].violations


def test_preflight_reports_missing_and_present_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "tau-0-wm"
    repo.mkdir()
    config = repo / "configs" / "deployment" / "wan_pretrain_rela_eef6d.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("action_dim: 20\n", encoding="utf-8")
    (repo / "run_infer_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    report = run_preflight(
        tau_repo=repo,
        tau_checkpoint=tmp_path / "missing_tau",
        wan_root=tmp_path / "missing_wan",
        require_cuda=False,
    )

    assert report["checks"]["tau_repo"]["ok"] is True
    assert report["checks"]["deployment_config"]["ok"] is True
    assert report["checks"]["tau_checkpoint"]["ok"] is False
    assert report["ready"] is False

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["ready"] is False


def test_runtime_recorder_writes_latency_and_memory_report(tmp_path: Path) -> None:
    recorder = RuntimeRecorder()
    recorder.add_sample(name="mock_infer", latency_s=0.25, gpu_memory_mb=1024.0)
    recorder.add_sample(name="mock_infer", latency_s=0.50, gpu_memory_mb=2048.0)

    report = recorder.summary()

    assert report["samples"] == 2
    assert report["latency_s"]["mean"] == 0.375
    assert report["gpu_memory_mb"]["max"] == 2048.0

    out = tmp_path / "runtime.json"
    recorder.write_json(out)
    assert json.loads(out.read_text(encoding="utf-8"))["samples"] == 2


def test_acvs_status_reports_missing_simulator_assets(tmp_path: Path) -> None:
    tau_repo = tmp_path / "tau-0-wm"
    tau_repo.mkdir()

    status = inspect_acvs_readiness(tau_repo=tau_repo, simulator_checkpoint=tmp_path / "missing_sim")

    assert status["ready"] is False
    assert status["checks"]["simulator_checkpoint"]["ok"] is False
    assert "simulator_checkpoint" in status["missing"]
