from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

from .activation import run_activation
from .dmn import run_dmn
from .masks import get_group_mask
from .one_sample import run_one_sample
from .roi_lmm import run_roi_lmm
from ...decisions import load_configured_decisions

logger = logging.getLogger(__name__)

def run(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Run configured group analyses over existing subject-level derivatives."""

    group = config.get("group", {})
    if not isinstance(group, dict):
        raise ValueError("group configuration must be an object.")
    if dry_run:
        return {"success": True, "skipped": False, "message": "Group analysis dry run.", "details": group}

    results = {}
    group_mask = get_group_mask(config, _reference_image(config))
    one_sample = group.get("one_sample", {})
    if isinstance(one_sample, dict) and one_sample.get("enabled", False):
        decision_manifest = _decision_manifest(config, one_sample, "group.one_sample")
        one_sample_kwargs = {"decision_manifest": decision_manifest}
        if group_mask is not None:
            one_sample_kwargs["mask_img"] = group_mask
        results["one_sample"] = run_one_sample(config, one_sample, **one_sample_kwargs)
    activation = group.get("activation", {})
    if isinstance(activation, dict) and activation.get("enabled", False):
        activation_kwargs = {
            "decision_manifest": _decision_manifest(config, activation, "group.activation")
        }
        if group_mask is not None:
            activation_kwargs["mask_img"] = group_mask
        results["activation"] = run_activation(config, activation, **activation_kwargs)
    dmn = group.get("dmn", {})
    if isinstance(dmn, dict) and dmn.get("enabled", False):
        results["dmn"] = run_dmn(
            config,
            dmn,
            decision_manifest=_decision_manifest(config, dmn, "group.dmn"),
        )
    roi_lmm = group.get("roi_lmm", {})
    if isinstance(roi_lmm, dict) and roi_lmm.get("enabled", False):
        results["roi_lmm"] = run_roi_lmm(
            config,
            roi_lmm,
            decision_manifest=_decision_manifest(config, roi_lmm, "group.roi_lmm"),
        )
    return {
        "success": True,
        "skipped": not results,
        "message": "Group analyses completed." if results else "No group analyses are enabled.",
        "details": results,
    }


def _reference_image(config: dict[str, Any]) -> str | None:
    """Find the first subject-level effect map for the shared mask grid."""

    output_dir = config.get("analysis", {}).get("output_dir")
    if not output_dir:
        return None
    root = Path(output_dir)
    for path in sorted(root.rglob("*_stat-effect.nii.gz")):
        if "group" not in path.relative_to(root).parts:
            return str(path)
    return None


def _decision_manifest(
    config: dict[str, Any], specification: dict[str, Any], field_name: str
):
    decision_id = str(specification.get("decision", "")).strip()
    logger.info("decision_manifest function")
    if not decision_id:
        return None
    decisions = load_configured_decisions(config)
    if decision_id not in decisions:
        available = ", ".join(sorted(decisions)) or "none"
        raise ValueError(
            f"{field_name}.decision '{decision_id}' is not registered; available decisions: {available}."
        )
    manifest = decisions[decision_id]
    print(
        f"[group] {field_name} loaded decision manifest '{manifest.analysis_id}' "
        f"from {manifest.path} (sha256={manifest.content_hash[:12]}...)"
    )
    return manifest
