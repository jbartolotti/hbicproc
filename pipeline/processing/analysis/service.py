from __future__ import annotations

import logging
from typing import Any

from .dataset import DatasetIndex
from .models import CanonicalGLMModel
from .planning import build_task_plans

logger = logging.getLogger(__name__)


def run(
    subject: str,
    config: dict[str, Any],
    dry_run: bool = False,
    rerun: bool = False,
) -> dict[str, Any]:
    """Plan and fit configured canonical GLMs independently for each run."""

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
    fitted_models = []
    errors = []
    for task_plan in task_plans:
        planned_analyses.extend(plan.as_dict() for plan in task_plan.analysis_plans(subject, output_root))
        dataset = DatasetIndex.from_config(config, task_config={"input_dataset": task_plan.input_dataset.as_dict()})
        runs = dataset.get_task_runs(subject=subject, task=task_plan.task, input_dataset=task_plan.input_dataset)
        contexts = runs or [
            # Preserve useful plan output when inputs are not present (for dry runs
            # and configuration validation); no fit is attempted for this context.
            task_plan.model_plans(subject, output_root)[0].context
            if task_plan.models
            else None
        ]
        contexts = [context for context in contexts if context is not None]
        for context in contexts:
            for model_spec in task_plan.models:
                model_plan = model_spec.plan(context, output_root)
                planned_models.append(model_plan.as_dict())
                if dry_run or not runs:
                    continue
                if model_spec.model_type != "canonical_glm":
                    errors.append(
                        f"Unsupported model type '{model_spec.model_type}' for model '{model_spec.name}'."
                    )
                    continue
                try:
                    model = CanonicalGLMModel(model_spec)
                    fitted = model.fit(model_plan)
                    metadata_path = model.write_metadata(fitted)
                    statistics_path = model.write_sufficient_statistics(fitted)
                    fitted_models.append({
                        "model": model_spec.name,
                        "context": context.as_dict(),
                        "metadata": str(metadata_path),
                        "sufficient_statistics": str(statistics_path),
                    })
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append(
                        f"{model_spec.name} ({context.subject}/{context.task}/"
                        f"{context.session or 'noses'}/{context.run or 'norun'}): {exc}"
                    )

    logger.info(
        "Prepared analysis plan for subject=%s tasks=%s models=%d analyses=%d",
        subject,
        [task.task for task in task_plans],
        len(planned_models),
        len(planned_analyses),
    )
    execution_mode = "dry run" if dry_run else "model fitting"
    rerun_text = " with model rerun requested" if rerun else ""
    success = not errors
    return {
        "success": success,
        "skipped": False,
        "message": (
            f"Prepared analysis plan for subject {subject} ({execution_mode}{rerun_text}); "
            f"fitted {len(fitted_models)} model run(s)."
            if success
            else f"Analysis failed for subject {subject}: {'; '.join(errors)}"
        ),
        "details": {
            "subject": subject,
            "tasks": [task.as_dict() for task in task_plans],
            "models": planned_models,
            "analyses": planned_analyses,
            "fitted_models": fitted_models,
            "errors": errors,
        },
    }


__all__ = ["run"]
