"""Supervision mask builder for Apex-WAM-Mini (v3.2 §10.4)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from registry import CompatibilityRegistry


def build_supervision_mask(
    sample: Dict[str, Any],
    registry: CompatibilityRegistry,
) -> Dict[str, bool]:
    """Return per-modality supervision flags for one training sample."""
    source_id = str(sample.get("source_id", ""))
    actions = sample.get("actions")
    spec = registry.lookup(source_id)

    action_compatible = (
        actions is not None
        and spec is not None
        and spec.action_supervision
        and spec.verified
    )
    video_compatible = not registry.is_video_only(source_id) or source_id in registry.compatible_sources

    return {
        "video": bool(video_compatible),
        "action": bool(action_compatible),
        "success": sample.get("success") is not None,
    }


def annotate_sample_metadata(
    *,
    source_id: str,
    registry: CompatibilityRegistry,
    has_actions: bool,
    success: Optional[bool] = None,
) -> Dict[str, Any]:
    """Attach v3.2 sample fields used by the dataloader."""
    spec = registry.lookup(source_id)
    return {
        "source_id": source_id,
        "action_space": spec.action_space if spec else "unknown",
        "embodiment_id": spec.embodiment if spec else "unknown",
        "success": success,
        "actions": None if not has_actions else "present",
        "supervision_mask": build_supervision_mask(
            {"source_id": source_id, "actions": "x" if has_actions else None, "success": success},
            registry,
        ),
    }
