from __future__ import annotations

from typing import Any

from ..behavior.workflow import run_behavior_events


def run_behavior_step(subject: str, config: dict[str, Any], task_name: str | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    return run_behavior_events(subject, config, task_name=task_name, dry_run=dry_run)
