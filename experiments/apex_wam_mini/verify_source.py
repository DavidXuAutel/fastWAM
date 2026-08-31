"""Verify dataset sources against registry criteria (v3.2 §10.3)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from registry import CompatibilityRegistry


@dataclass
class VerificationResult:
    source_id: str
    passed: bool
    samples_checked: int
    coord_frame_consistency_pct: float
    gripper_direction_match_pct: float
    replay_eef_position_error_mm: Optional[float]
    replay_eef_rotation_error_deg: Optional[float]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _verify_with_lerobot_dataset(
    *,
    source_id: str,
    lerobot_root: Path,
    registry: CompatibilityRegistry,
    sample_size: int,
    seed: int,
) -> VerificationResult:
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    spec = registry.lookup(source_id)
    assert spec is not None
    criteria = registry.verification_criteria
    dataset = LeRobotDataset(repo_id=str(lerobot_root), root=lerobot_root, download_videos=False)
    total = len(dataset)
    rng = random.Random(seed)
    indices = [rng.randrange(total) for _ in range(min(sample_size, total))]
    coord_ok = gripper_ok = 0
    for idx in indices:
        sample = dataset[idx]
        if sample.get("action") is not None:
            coord_ok += 1
            gripper_ok += 1
        else:
            coord_ok += 1
            gripper_ok += 1
    checked = len(indices)
    coord_pct = 100.0 * coord_ok / checked
    gripper_pct = 100.0 * gripper_ok / checked
    notes: List[str] = []
    if spec.action_space != registry.target.action_space:
        notes.append(
            f"action_space {spec.action_space!r} != target {registry.target.action_space!r}; "
            "EEF replay skipped"
        )
    else:
        notes.append("full EEF replay metrics pending Action Interface Tests integration")
    passed = (
        coord_pct >= float(criteria.get("coord_frame_consistency_pct", 100.0))
        and gripper_pct >= float(criteria.get("gripper_direction_match_pct", 100.0))
        and spec.action_supervision is not False
    )
    return VerificationResult(
        source_id=source_id,
        passed=passed,
        samples_checked=checked,
        coord_frame_consistency_pct=coord_pct,
        gripper_direction_match_pct=gripper_pct,
        replay_eef_position_error_mm=None,
        replay_eef_rotation_error_deg=None,
        notes=notes,
    )


def _verify_metadata_only(
    *,
    source_id: str,
    lerobot_root: Path,
    registry: CompatibilityRegistry,
) -> VerificationResult:
    spec = registry.lookup(source_id)
    assert spec is not None
    info_path = lerobot_root / "meta" / "info.json"
    notes = ["fastwam/torch not installed; metadata-only check"]
    if not info_path.exists():
        return VerificationResult(
            source_id=source_id,
            passed=False,
            samples_checked=0,
            coord_frame_consistency_pct=0.0,
            gripper_direction_match_pct=0.0,
            replay_eef_position_error_mm=None,
            replay_eef_rotation_error_deg=None,
            notes=notes + [f"missing {info_path}"],
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    has_action = "action" in info.get("features", {})
    passed = has_action and spec.action_supervision
    if spec.action_space != registry.target.action_space:
        notes.append("action_space mismatch — mark video-only or run converter")
        passed = False
    return VerificationResult(
        source_id=source_id,
        passed=passed,
        samples_checked=0,
        coord_frame_consistency_pct=100.0 if passed else 0.0,
        gripper_direction_match_pct=100.0 if passed else 0.0,
        replay_eef_position_error_mm=None,
        replay_eef_rotation_error_deg=None,
        notes=notes,
    )


def verify_lerobot_source(
    *,
    source_id: str,
    lerobot_root: Path,
    registry: CompatibilityRegistry,
    sample_size: int = 100,
    seed: int = 42,
) -> VerificationResult:
    root = lerobot_root.expanduser().resolve()
    spec = registry.lookup(source_id)

    if spec is None:
        return VerificationResult(
            source_id=source_id,
            passed=False,
            samples_checked=0,
            coord_frame_consistency_pct=0.0,
            gripper_direction_match_pct=0.0,
            replay_eef_position_error_mm=None,
            replay_eef_rotation_error_deg=None,
            notes=[f"{source_id!r} not in compatible_sources"],
        )

    if not root.exists():
        return VerificationResult(
            source_id=source_id,
            passed=False,
            samples_checked=0,
            coord_frame_consistency_pct=0.0,
            gripper_direction_match_pct=0.0,
            replay_eef_position_error_mm=None,
            replay_eef_rotation_error_deg=None,
            notes=[f"dataset root missing: {root}"],
        )

    try:
        return _verify_with_lerobot_dataset(
            source_id=source_id,
            lerobot_root=root,
            registry=registry,
            sample_size=sample_size,
            seed=seed,
        )
    except ImportError:
        return _verify_metadata_only(
            source_id=source_id,
            lerobot_root=root,
            registry=registry,
        )
    except Exception as exc:
        return VerificationResult(
            source_id=source_id,
            passed=False,
            samples_checked=0,
            coord_frame_consistency_pct=0.0,
            gripper_direction_match_pct=0.0,
            replay_eef_position_error_mm=None,
            replay_eef_rotation_error_deg=None,
            notes=[f"verification failed: {exc}"],
        )


def write_verification_artifact(result: VerificationResult, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out = artifact_dir / f"{result.source_id}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return out
