from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .analyses import AnalysisPlan, AnalysisSpec
from .context import InputDataset, TaskRunContext
from .models import ModelPlan, ModelSpec


@dataclass(frozen=True)
class TaskPlan:
    """Configuration plan for one enabled task.

    A task plan contains specifications only. It does not discover files, fit a
    model, or execute an analysis.
    """

    task: str
    enabled: bool
    input_dataset: InputDataset
    models: tuple[ModelSpec, ...]
    analyses: tuple[AnalysisSpec, ...]

    def model_plans(self, subject: str, output_root: str | Path) -> tuple[ModelPlan, ...]:
        context = TaskRunContext(subject=subject, task=self.task, input_dataset=self.input_dataset)
        return tuple(model.plan(context, str(output_root)) for model in self.models)

    def analysis_plans(self, subject: str, output_root: str | Path) -> tuple[AnalysisPlan, ...]:
        context = TaskRunContext(subject=subject, task=self.task, input_dataset=self.input_dataset)
        models = {model.name: model for model in self.models}
        return tuple(
            analysis.plan(context, models[analysis.model], str(output_root))
            for analysis in self.analyses
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "enabled": self.enabled,
            "input_dataset": self.input_dataset.as_dict(),
            "models": [
                {
                    "name": model.name,
                    "type": model.model_type,
                    "configuration": dict(model.configuration),
                }
                for model in self.models
            ],
            "analyses": [
                {
                    "name": analysis.name,
                    "model": analysis.model,
                    "configuration": dict(analysis.configuration),
                    "required_derivatives": list(analysis.required_derivatives),
                }
                for analysis in self.analyses
            ],
        }


def _input_dataset(raw: Mapping[str, Any], *, field_name: str) -> InputDataset:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object with 'name' and 'path'.")

    name = str(raw.get("name", "")).strip()
    path_value = str(raw.get("path", "")).strip()
    if not name or not path_value:
        raise ValueError(f"{field_name} must define non-empty 'name' and 'path' values.")
    return InputDataset(name=name, path=Path(path_value))


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    return value


def build_task_plans(config: Mapping[str, Any], subject: str) -> tuple[TaskPlan, ...]:
    """Build task/model/analysis plans from the authoritative JSON schema."""

    analysis_config = _mapping(config.get("analysis"), field_name="analysis")
    global_dataset = _input_dataset(
        _mapping(analysis_config.get("input_dataset"), field_name="analysis.input_dataset"),
        field_name="analysis.input_dataset",
    )
    subject_config = _mapping(analysis_config.get("subject"), field_name="analysis.subject")
    task_configs = _mapping(subject_config.get("tasks"), field_name="analysis.subject.tasks")
    output_root = str(analysis_config.get("output_dir", "")).strip()
    if not output_root:
        raise ValueError("analysis.output_dir must not be empty.")

    plans: list[TaskPlan] = []
    for task_name, raw_task_config in task_configs.items():
        task = str(task_name).strip()
        if not task:
            raise ValueError("Task names must not be empty.")
        task_config = _mapping(raw_task_config, field_name=f"analysis.subject.tasks.{task}")
        if not bool(task_config.get("enabled", True)):
            continue

        dataset = global_dataset
        if "input_dataset" in task_config:
            dataset = _input_dataset(
                _mapping(
                    task_config.get("input_dataset"),
                    field_name=f"analysis.subject.tasks.{task}.input_dataset",
                ),
                field_name=f"analysis.subject.tasks.{task}.input_dataset",
            )

        model_configs = _mapping(
            task_config.get("models"),
            field_name=f"analysis.subject.tasks.{task}.models",
        )
        models: list[ModelSpec] = []
        for model_name, raw_model_config in model_configs.items():
            model = str(model_name).strip()
            model_config = _mapping(
                raw_model_config,
                field_name=f"analysis.subject.tasks.{task}.models.{model}",
            )
            model_type = str(model_config.get("type", "")).strip()
            if not model_type:
                raise ValueError(
                    f"Model '{model}' for task '{task}' must define a non-empty 'type'."
                )
            models.append(
                ModelSpec(name=model, model_type=model_type, configuration=dict(model_config))
            )

        model_names = {model.name for model in models}
        analysis_configs = _mapping(
            task_config.get("analyses"),
            field_name=f"analysis.subject.tasks.{task}.analyses",
        )
        analyses: list[AnalysisSpec] = []
        for analysis_name, raw_analysis_config in analysis_configs.items():
            analysis_name_text = str(analysis_name).strip()
            analysis_config_value = _mapping(
                raw_analysis_config,
                field_name=f"analysis.subject.tasks.{task}.analyses.{analysis_name_text}",
            )
            if not bool(analysis_config_value.get("enabled", True)):
                continue
            model_name = str(analysis_config_value.get("model", "")).strip()
            if not model_name:
                raise ValueError(
                    f"Analysis '{analysis_name_text}' for task '{task}' must reference a model."
                )
            if model_name not in model_names:
                raise ValueError(
                    f"Analysis '{analysis_name_text}' for task '{task}' references unknown model '{model_name}'."
                )
            analyses.append(
                AnalysisSpec(
                    name=analysis_name_text,
                    model=model_name,
                    configuration=dict(analysis_config_value),
                )
            )

        plans.append(
            TaskPlan(
                task=task,
                enabled=True,
                input_dataset=dataset,
                models=tuple(models),
                analyses=tuple(analyses),
            )
        )

    _ = subject
    _ = output_root
    return tuple(plans)


__all__ = ["TaskPlan", "build_task_plans"]
