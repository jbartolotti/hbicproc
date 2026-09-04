from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel

from ..context import TaskRunContext
from .base import ModelPlan, ModelSpec


@dataclass(frozen=True)
class FittedModel:
    """A fitted canonical first-level model and its execution metadata."""

    plan: ModelPlan
    estimator: FirstLevelModel
    events: pd.DataFrame
    confounds: pd.DataFrame | None


class CanonicalGLMModel:
    """Fit one canonical first-level GLM for one task run."""

    model_type = "canonical_glm"

    def __init__(self, spec: ModelSpec) -> None:
        if spec.model_type != self.model_type:
            raise ValueError(
                f"Model '{spec.name}' has type '{spec.model_type}', expected '{self.model_type}'."
            )
        self.spec = spec

    def fit(self, plan: ModelPlan) -> FittedModel:
        context = plan.context
        if context.bold_path is None or context.events_path is None:
            raise ValueError("Canonical GLM fitting requires BOLD and events paths.")
        if not context.bold_path.exists():
            raise FileNotFoundError(f"BOLD input does not exist: {context.bold_path}")
        if not context.events_path.exists():
            raise FileNotFoundError(f"Events input does not exist: {context.events_path}")

        events = pd.read_csv(context.events_path, sep="\t")
        required_event_columns = {"onset", "duration", "trial_type"}
        missing_events = sorted(required_event_columns - set(events.columns))
        if missing_events:
            raise ValueError(
                f"Events file {context.events_path} is missing columns: {', '.join(missing_events)}."
            )

        confounds = self._load_confounds(context)
        configuration = dict(self.spec.configuration)
        estimator = FirstLevelModel(
            t_r=self._repetition_time(context.bold_path, configuration),
            hrf_model=configuration.get("hrf_model", "glover"),
            drift_model=configuration.get("drift_model", "cosine"),
            high_pass=configuration.get("high_pass", 0.01),
            smoothing_fwhm=configuration.get("smoothing_fwhm"),
            standardize=configuration.get("standardize", False),
            noise_model=configuration.get("noise_model", "ar1"),
            minimize_memory=False,
            reports=False,
            n_jobs=configuration.get("n_jobs", 1),
        )
        estimator.fit(
            str(context.bold_path),
            events=events,
            confounds=confounds,
        )
        return FittedModel(plan=plan, estimator=estimator, events=events, confounds=confounds)

    def write_metadata(self, fitted: FittedModel) -> Path:
        output_dir = fitted.plan.derivatives.model_directory
        if output_dir is None:
            raise ValueError("Canonical GLM model plan has no model derivative directory.")
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "model_metadata.json"
        metadata = {
            "model": self.spec.name,
            "model_type": self.spec.model_type,
            "context": fitted.plan.context.as_dict(),
            "fingerprint_configuration": self.spec.fingerprint_configuration,
            "design_columns": list(fitted.estimator.design_matrices_[0].columns),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        return metadata_path

    def write_sufficient_statistics(self, fitted: FittedModel) -> Path:
        """Persist beta/covariance statistics needed for later contrasts."""

        output_dir = fitted.plan.derivatives.model_directory
        if output_dir is None:
            raise ValueError("Canonical GLM model plan has no model derivative directory.")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = fitted.estimator.results_
        arrays: dict[str, np.ndarray] = {
            "design_matrix": np.asarray(fitted.estimator.design_matrices_[0], dtype=float),
        }
        result_metadata: list[dict[str, Any]] = []
        for index, result_group in enumerate(results):
            arrays[f"theta_{index}"] = np.asarray(
                np.stack([np.asarray(result.theta) for result in result_group.values()])
            )
            arrays[f"cov_{index}"] = np.asarray(
                np.stack([np.asarray(result.cov) for result in result_group.values()])
            )
            arrays[f"dispersion_{index}"] = np.asarray(
                np.stack([np.asarray(result.dispersion) for result in result_group.values()])
            )
            result_metadata.append({
                "labels": [str(label) for label in result_group],
                "df_residuals": [int(result.df_residuals) for result in result_group.values()],
            })

        statistics_path = output_dir / "sufficient_statistics.npz"
        np.savez_compressed(statistics_path, **arrays)
        metadata_path = output_dir / "sufficient_statistics.json"
        metadata = {
            "version": 1,
            "model": self.spec.name,
            "model_type": self.spec.model_type,
            "fingerprint_configuration": self.spec.fingerprint_configuration,
            "context": fitted.plan.context.as_dict(),
            "design_columns": list(fitted.estimator.design_matrices_[0].columns),
            "result_groups": result_metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return statistics_path

    def _load_confounds(self, context: TaskRunContext) -> pd.DataFrame | None:
        configuration = self.spec.configuration
        requested = configuration.get("confounds")
        if not requested:
            return None
        if context.confounds_path is None or not context.confounds_path.exists():
            raise FileNotFoundError("Configured confounds require an existing confounds file.")

        confounds = pd.read_csv(context.confounds_path, sep="\t")
        columns = self._confound_columns(requested, confounds.columns)
        return confounds.loc[:, columns] if columns else confounds

    @staticmethod
    def _confound_columns(requested: Any, available: pd.Index) -> list[str]:
        available_names = [str(column) for column in available]
        if requested is True or requested == "all":
            return available_names
        if isinstance(requested, Mapping):
            requested = [key for key, enabled in requested.items() if enabled]
        if isinstance(requested, str):
            requested = [requested]
        if not isinstance(requested, (list, tuple, set)):
            return []
        return [str(column) for column in requested if str(column) in available_names]

    @staticmethod
    def _repetition_time(bold_path: Path, configuration: Mapping[str, Any]) -> float:
        configured = configuration.get("t_r")
        if configured is not None:
            return float(configured)
        image = nib.load(str(bold_path))
        if len(image.header.get_zooms()) < 4 or image.header.get_zooms()[3] <= 0:
            raise ValueError(f"Could not determine a positive repetition time from {bold_path}.")
        return float(image.header.get_zooms()[3])


__all__ = ["CanonicalGLMModel", "FittedModel"]
