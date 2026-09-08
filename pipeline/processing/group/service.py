from __future__ import annotations

from typing import Any

from .activation import run_activation
from .dmn import run_dmn


def run(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Run configured group analyses over existing subject-level derivatives."""

    group = config.get("group", {})
    if not isinstance(group, dict):
        raise ValueError("group configuration must be an object.")
    if dry_run:
        return {"success": True, "skipped": False, "message": "Group analysis dry run.", "details": group}

    results = {}
    activation = group.get("activation", {})
    if isinstance(activation, dict) and activation.get("enabled", False):
        results["activation"] = run_activation(config, activation)
    dmn = group.get("dmn", {})
    if isinstance(dmn, dict) and dmn.get("enabled", False):
        results["dmn"] = run_dmn(config, dmn)
    return {
        "success": True,
        "skipped": not results,
        "message": "Group analyses completed." if results else "No group analyses are enabled.",
        "details": results,
    }
