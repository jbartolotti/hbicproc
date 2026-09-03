from __future__ import annotations

import logging
from typing import Any

from .planning import build_task_plans

logger = logging.getLogger(__name__)


def run(
    subject: str,
    config: dict[str, Any],
    dry_run: bool = False,
    rerun: bool = False,
) -> dict[str, Any]:
    """Build the PR 1 analysis plan without fitting or executing models."""

    try:
        task_plans = build_task_plans(config, subject)
    except (TypeError, ValueError) as exc:
        logger.warning("Analysis configuration is invalid for subject=%s: %s", subject, exc)
        return {
            "success": False,
            "skipped": False,
            "message": f"Analysis configuration is invalid for subject {subject}: {exc}",
            "details": {"subject": subject, "error": str(exc)},
        }

    if not task_plans:
        return {
            "success": True,
            "skipped": True,
            "message": f"No enabled analysis tasks are configured for subject {subject}.",
            "details": {"subject": subject, "tasks": []},
        }

    analysis_config = config.get("analysis", {})
    output_root = analysis_config.get("output_dir", "") if isinstance(analysis_config, dict) else ""
    planned_models = []
    planned_analyses = []
    for task_plan in task_plans:
        planned_models.extend(plan.as_dict() for plan in task_plan.model_plans(subject, output_root))
        planned_analyses.extend(plan.as_dict() for plan in task_plan.analysis_plans(subject, output_root))

    logger.info(
        "Prepared analysis plan for subject=%s tasks=%s models=%d analyses=%d",
        subject,
        [task.task for task in task_plans],
        len(planned_models),
        len(planned_analyses),
    )
    execution_mode = "dry run" if dry_run else "planning only"
    rerun_text = " with model rerun requested" if rerun else ""
    return {
        "success": True,
        "skipped": False,
        "message": (
            f"Prepared analysis plan for subject {subject} ({execution_mode}{rerun_text}); "
            "no model execution was performed."
        ),
        "details": {
            "subject": subject,
            "tasks": [task.as_dict() for task in task_plans],
            "models": planned_models,
            "analyses": planned_analyses,
        },
    }


__all__ = ["run"]
