"""Apex-WAM-Mini data compatibility registry (multi-profile)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from yaml_utils import load_yaml, save_yaml


@dataclass(frozen=True)
class SourceSpec:
    id: str
    profiles: List[str]
    action_space: str
    embodiment: str
    coord_frame: str
    action_dim: int
    action_supervision: bool
    verified: bool
    notes: str = ""


@dataclass(frozen=True)
class TargetSpec:
    action_space: str
    embodiment: str
    coord_frame: str
    action_dim: int
    deploy_action_dim: int = 16


@dataclass
class CompatibilityRegistry:
    profile: str
    target: TargetSpec
    compatible_sources: Dict[str, SourceSpec]
    video_only_sources: List[str]
    verification_criteria: Dict[str, Any]
    _raw: Dict[str, Any]

    @classmethod
    def load(cls, path: Path | str, profile: Optional[str] = None) -> "CompatibilityRegistry":
        raw = load_yaml(path)
        active = profile or raw.get("default_profile", "g1")
        profiles = raw.get("profiles", {})
        if active not in profiles:
            raise KeyError(f"unknown profile {active!r}; available: {list(profiles)}")

        t = profiles[active]["target"]
        target = TargetSpec(
            action_space=str(t["action_space"]),
            embodiment=str(t["embodiment"]),
            coord_frame=str(t["coord_frame"]),
            action_dim=int(t["action_dim"]),
            deploy_action_dim=int(t.get("deploy_action_dim", t["action_dim"])),
        )

        compatible: Dict[str, SourceSpec] = {}
        for item in raw.get("compatible_sources", []):
            item_profiles = [str(p) for p in item.get("profiles", [active])]
            if active not in item_profiles:
                continue
            spec = SourceSpec(
                id=str(item["id"]),
                profiles=item_profiles,
                action_space=str(item["action_space"]),
                embodiment=str(item["embodiment"]),
                coord_frame=str(item["coord_frame"]),
                action_dim=int(item["action_dim"]),
                action_supervision=bool(item.get("action_supervision", True)),
                verified=bool(item.get("verified", False)),
                notes=str(item.get("notes", "")),
            )
            compatible[spec.id] = spec

        video_map = raw.get("video_only_by_profile", {})
        legacy = raw.get("video_only_sources", [])
        if active in video_map:
            video_only = [str(x) for x in video_map[active]]
        else:
            video_only = [str(x) for x in legacy]

        criteria = raw.get("verification_criteria", {})
        return cls(
            profile=active,
            target=target,
            compatible_sources=compatible,
            video_only_sources=video_only,
            verification_criteria=criteria,
            _raw=raw,
        )

    def lookup(self, source_id: str) -> Optional[SourceSpec]:
        return self.compatible_sources.get(source_id)

    def is_video_only(self, source_id: str) -> bool:
        if source_id in self.compatible_sources:
            spec = self.compatible_sources[source_id]
            if spec.action_supervision and spec.verified:
                return False
        return source_id in self.video_only_sources or (
            source_id in self.compatible_sources
            and not self.compatible_sources[source_id].action_supervision
        )

    def allows_action_supervision(self, source_id: str) -> bool:
        spec = self.lookup(source_id)
        if spec is None:
            return False
        return spec.action_supervision and spec.verified

    def save_verified_flag(self, registry_path: Path, source_id: str, verified: bool) -> None:
        for item in self._raw.get("compatible_sources", []):
            if str(item["id"]) == source_id:
                item["verified"] = verified
                break
        else:
            raise KeyError(f"source_id {source_id!r} not in compatible_sources")
        save_yaml(registry_path, self._raw)


def default_registry_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / "configs" / "data_compatibility.yaml"
