from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
from nibabel.dft import logger
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel

from ..context import TaskRunContext
from ..derivatives import DerivativePathBuilder
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


        img = nib.load(str(context.bold_path))

        logger.info(
            "GLM input check: file=%s volumes=%d confounds_shape=%s events_rows=%d",
            context.bold_path,
            img.shape[-1],
            None if confounds is None else confounds.shape,
            len(events),
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
        metadata_path = self._model_file(fitted, desc="model-metadata", suffix=".json")
        metadata = {
            "model": self.spec.name,
            "model_type": self.spec.model_type,
            "context": fitted.plan.context.as_dict(),
            "fingerprint_configuration": self.spec.fingerprint_configuration,
            "design_columns": list(fitted.estimator.design_matrices_[0].columns),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        return metadata_path

    def write_design_matrix(self, fitted: FittedModel) -> Path:
        """Persist the fitted run's design matrix as a portable derivative."""

        output_dir = fitted.plan.derivatives.model_directory
        if output_dir is None:
            raise ValueError("Canonical GLM model plan has no model derivative directory.")
        output_dir.mkdir(parents=True, exist_ok=True)
        design_matrix_path = self._model_file(fitted, desc="design-matrix", suffix=".tsv")
        fitted.estimator.design_matrices_[0].to_csv(design_matrix_path, sep="\t", index=False)
        return design_matrix_path

    def write_derivatives(self, fitted: FittedModel) -> dict[str, list[str]]:
        """Write portable first-level products while the fitted model is in memory."""

        output_dir = fitted.plan.derivatives.model_directory
        if output_dir is None:
            raise ValueError("Canonical GLM model plan has no model derivative directory.")
        output_dir.mkdir(parents=True, exist_ok=True)
        context = fitted.plan.context
        derivatives: dict[str, list[str]] = {}

        derivatives["design_matrix"] = [str(self.write_design_matrix(fitted))]
        design_png = self._model_file(
            fitted,
            subject=context.subject,
            session=context.session,
            task=context.task,
            run=context.run,
            desc="design-matrix",
            suffix=".png",
        )
        design_png.parent.mkdir(parents=True, exist_ok=True)
        figure = fitted.estimator.design_matrices_[0].T
        plt.figure(figsize=(max(8, figure.shape[1] / 8), max(4, figure.shape[0] / 2)))
        plt.imshow(figure, aspect="auto", interpolation="nearest", cmap="viridis")
        plt.yticks(range(len(figure.index)), figure.index)
        plt.xlabel("Scan")
        plt.tight_layout()
        plt.savefig(design_png, dpi=150)
        plt.close()
        derivatives["design_matrix_png"] = [str(design_png)]

        mask_img = getattr(fitted.estimator, "mask_img_", None)
        if mask_img is not None:
            mask_path = self._model_file(
                fitted,
                subject=context.subject,
                session=context.session,
                task=context.task,
                run=context.run,
                desc="mask",
            )
            mask_img.to_filename(mask_path)
            derivatives["mask"] = [str(mask_path)]

        conditions = self.spec.configuration.get("conditions", ())
        if isinstance(conditions, str):
            conditions = [conditions]
        if isinstance(conditions, (list, tuple, set)):
            condition_paths = []
            for condition in conditions:
                condition_name = str(condition).strip()
                if not condition_name:
                    continue
                image = fitted.estimator.compute_contrast(condition_name, output_type="effect_size")
                condition_path = self._model_file(
                    fitted,
                    subject=context.subject,
                    session=context.session,
                    task=context.task,
                    run=context.run,
                    desc=condition_name,
                    stat="effect",
                )
                image.to_filename(condition_path)
                condition_paths.append(str(condition_path))
            if condition_paths:
                derivatives["condition_effects"] = condition_paths

        residuals = getattr(fitted.estimator, "residuals_", None)
        if residuals is None:
            residuals = getattr(fitted.estimator, "residuals", None)
        residual_paths = []
        if isinstance(residuals, (list, tuple)):
            for index, image in enumerate(residuals, start=1):
                if image is None or not hasattr(image, "to_filename"):
                    continue
                residual_path = self._model_file(
                    fitted,
                    subject=context.subject,
                    session=context.session,
                    task=context.task,
                    run=context.run,
                    desc=f"residual-{index:02d}",
                )
                image.to_filename(residual_path)
                residual_paths.append(str(residual_path))
        if residual_paths:
            derivatives["residuals"] = residual_paths

        if fitted.confounds is not None:
            motion_path = self._model_file(
                fitted,
                subject=context.subject,
                session=context.session,
                task=context.task,
                run=context.run,
                desc="motion-qc",
                suffix=".tsv",
            )
            fitted.confounds.to_csv(motion_path, sep="\t", index=False)
            summary_path = self._model_file(
                fitted,
                subject=context.subject,
                session=context.session,
                task=context.task,
                run=context.run,
                desc="motion-qc-summary",
                suffix=".json",
            )
            summary = {
                "columns": list(fitted.confounds.columns),
                "rows": int(len(fitted.confounds)),
                "mean": fitted.confounds.mean().to_dict(),
                "std": fitted.confounds.std().to_dict(),
                "maximum": fitted.confounds.max().to_dict(),
            }
            summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
            derivatives["motion_qc"] = [str(summary_path), str(motion_path)]

        generate_report = getattr(fitted.estimator, "generate_report", None)
        if callable(generate_report):
            report = generate_report()
            report_path = self._model_file(
                fitted,
                subject=context.subject,
                session=context.session,
                task=context.task,
                run=context.run,
                desc="first-level-report",
                suffix=".html",
            )
            report.save_as_html(str(report_path))
            derivatives["report"] = [str(report_path)]
        return derivatives

    def _model_file(
        self,
        fitted: FittedModel,
        *,
        desc: str,
        stat: str | None = None,
        suffix: str = ".nii.gz",
        **_: Any,
    ) -> Path:
        output_dir = fitted.plan.derivatives.model_directory
        if output_dir is None:
            raise ValueError("Canonical GLM model plan has no model derivative directory.")
        context = fitted.plan.context
        return output_dir / DerivativePathBuilder.build_filename(
            subject=context.subject,
            session=context.session,
            task=context.task,
            run=context.run,
            desc=desc,
            stat=stat,
            suffix=suffix,
        )

    def _load_confounds(self, context: TaskRunContext) -> pd.DataFrame | None:
        configuration = self.spec.configuration
        requested = configuration.get("confounds")
        if not requested:
            return None
        if context.confounds_path is None or not context.confounds_path.exists():
            raise FileNotFoundError("Configured confounds require an existing confounds file.")

        confounds = pd.read_csv(context.confounds_path, sep="\t")
        columns, missing = self._confound_columns(requested, confounds.columns)
        if missing:
            raise ValueError(
                f"Confounds file {context.confounds_path} is missing required columns: "
                f"{', '.join(missing)}."
            )
        selected = [column for column in columns if column in confounds.columns]
        if not selected:
            return None
        result = confounds.loc[:, selected].copy()
        try:
            result = result.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Confounds file {context.confounds_path} contains non-numeric required columns."
            ) from exc
        if result.isna().any().any():
            result = result.fillna(0)
        return result

    @staticmethod
    def _confound_columns(requested: Any, available: pd.Index) -> tuple[list[str], list[str]]:
        available_names = [str(column) for column in available]
        if requested is True or requested == "all":
            return available_names, []
        if isinstance(requested, Mapping):
            expanded: list[str] = []
            missing: list[str] = []
            for group, enabled in requested.items():
                if not enabled:
                    continue
                group_name = str(group)
                if group_name == "columns":
                    if not isinstance(enabled, (list, tuple, set)):
                        raise ValueError("The confounds.columns entry must be a list of column names.")
                    expanded.extend(str(column) for column in enabled)
                elif group_name == "motion":
                    expanded.extend([
                        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
                    ])
                elif group_name == "motion_derivatives":
                    expanded.extend([
                        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
                        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
                    ])
                elif group_name == "framewise_displacement":
                    expanded.append(group_name)
                elif group_name == "acompcor":
                    if not isinstance(enabled, int) or isinstance(enabled, bool) or enabled < 1:
                        raise ValueError("The acompcor confound group must be a positive integer.")
                    acompcor = sorted(column for column in available_names if column.startswith("a_comp_cor_"))
                    expanded.extend(acompcor[:enabled])
                    if len(acompcor) < enabled:
                        missing.extend([f"a_comp_cor_{index:02d}" for index in range(len(acompcor), enabled)])
                else:
                    expanded.append(group_name)
            requested = expanded
        else:
            missing = []
        if isinstance(requested, str):
            requested = [requested]
        if not isinstance(requested, (list, tuple, set)):
            return [], []
        requested_names = list(dict.fromkeys(str(column) for column in requested))
        available_set = set(available_names)
        return (
            [column for column in requested_names if column in available_set],
            missing + [column for column in requested_names if column not in available_set],
        )

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
