"""Scan LeRobot roots and build stage manifests for Apex-WAM-Mini."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Optional

from yaml_utils import load_yaml


@dataclass
class DatasetStats:
    source_id: str
    lerobot_root: str
    exists: bool
    fps: Optional[int] = None
    total_episodes: Optional[int] = None
    total_frames: Optional[int] = None
    hours: Optional[float] = None
    error: Optional[str] = None


@dataclass
class StageManifest:
    stage: str
    mix_weights: Dict[str, float]
    buckets: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    totals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_lerobot_info(root: Path) -> Dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _scan_lerobot_root(source_id: str, root: Path) -> DatasetStats:
    root = root.expanduser().resolve()
    if not root.exists():
        return DatasetStats(source_id=source_id, lerobot_root=str(root), exists=False)
    try:
        info = _read_lerobot_info(root)
        fps = int(info.get("fps", 0))
        frames = int(info.get("total_frames", 0))
        episodes = int(info.get("total_episodes", 0))
        hours = frames / fps / 3600.0 if fps > 0 else None
        return DatasetStats(
            source_id=source_id,
            lerobot_root=str(root),
            exists=True,
            fps=fps,
            total_episodes=episodes,
            total_frames=frames,
            hours=hours,
        )
    except Exception as exc:
        return DatasetStats(
            source_id=source_id,
            lerobot_root=str(root),
            exists=False,
            error=str(exc),
        )


def scan_sources(sources_cfg: Dict[str, Any]) -> List[DatasetStats]:
    stats: List[DatasetStats] = []
    for source_id, entry in sources_cfg.get("sources", {}).items():
        roots = entry.get("lerobot_roots", [])
        for root in roots:
            stats.append(_scan_lerobot_root(source_id, Path(root)))
    return stats


def _resolve_category(entry: Dict[str, Any], profile: str) -> str:
    if "category_by_profile" in entry:
        by_prof = entry["category_by_profile"]
        if profile in by_prof:
            return str(by_prof[profile])
    profiles = entry.get("profiles", [])
    if profiles and profile not in profiles:
        return ""
    return str(entry.get("category", "video_only"))


def _bucket_sources(sources_cfg: Dict[str, Any], profile: str) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {}
    for source_id, entry in sources_cfg.get("sources", {}).items():
        category = _resolve_category(entry, profile)
        if not category:
            continue
        buckets.setdefault(category, []).append(source_id)
    return buckets


MIX_BUCKET_ALIASES = {
    "target_action": "target_action_data",
    "compatible_action": "compatible_action_data",
    "video_only": "video_only_or_incompatible",
    "failure": "failure",
}


def _mix_weight(mix_weights: Dict[str, float], bucket_name: str) -> float:
    key = MIX_BUCKET_ALIASES.get(bucket_name, bucket_name)
    if key not in mix_weights and bucket_name == "target_action":
        key = "robot_target"
    return float(mix_weights.get(key, mix_weights.get(bucket_name, 0.0)))


def _mix_key(stage: str, profile: str) -> str:
    stage = stage.upper()
    if profile == "franka":
        return f"franka_stage_{stage.lower()}_mix"
    return f"stage_{stage.lower()}_mix"


def build_stage_manifest(
    stage: str,
    sources_cfg: Dict[str, Any],
    scan: List[DatasetStats],
    profile: Optional[str] = None,
) -> StageManifest:
    active = profile or sources_cfg.get("active_profile", "g1")
    if stage.upper() not in ("B", "C"):
        raise ValueError(f"unsupported stage {stage!r}, expected B or C")

    mix_key = _mix_key(stage, active)
    if mix_key not in sources_cfg:
        raise KeyError(f"missing mix weights {mix_key} for profile {active!r}")

    mix_weights = dict(sources_cfg[mix_key])
    by_source = {s.source_id: s for s in scan}
    buckets_cfg = _bucket_sources(sources_cfg, active)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    total_hours = 0.0
    total_frames = 0

    for bucket_name, source_ids in buckets_cfg.items():
        entries = []
        for sid in source_ids:
            stat = by_source.get(sid)
            if stat is None:
                continue
            entry = asdict(stat)
            weight = _mix_weight(mix_weights, bucket_name) / max(len(source_ids), 1)
            entry["bucket"] = bucket_name
            entry["mix_weight_share"] = weight
            entries.append(entry)
            if stat.hours:
                total_hours += stat.hours * weight
            if stat.total_frames:
                total_frames += int(stat.total_frames * weight)
        if entries:
            buckets[bucket_name] = entries

    return StageManifest(
        stage=stage.upper(),
        mix_weights=mix_weights,
        buckets=buckets,
        totals={
            "profile": active,
            "weighted_hours_estimate": round(total_hours, 3),
            "weighted_frames_estimate": int(total_frames),
        },
    )


def write_manifest(manifest: StageManifest, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n")


def write_scan_report(scan: List[DatasetStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in scan]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_sources_config(path: Path) -> Dict[str, Any]:
    return load_yaml(path)
