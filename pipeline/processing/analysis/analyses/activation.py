from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nilearn.image import load_img

from ..models.canonical_glm import FittedModel
from ..derivatives import DerivativePathBuilder
from .atlas import AtlasCache
from .base import AnalysisPlan


@dataclass(frozen=True)
class ContrastOutput:
    """One contrast image generated from an in-memory fitted GLM."""

    contrast: str
    output_type: str
    path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "contrast": self.contrast,
            "output_type": self.output_type,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class AtlasOutput:
    """An atlas resampled to the fitted model's image space."""

    atlas: str
    path: Path

    def as_dict(self) -> dict[str, str]:
        return {"atlas": self.atlas, "path": str(self.path)}


@dataclass(frozen=True)
class ReportOutput:
    """HTML report generated from the fitted model and activation contrasts."""

    path: Path

    def as_dict(self) -> dict[str, str]:
        return {"report": str(self.path)}


class ActivationAnalysis:
    """Compute configured activation contrasts from a fitted first-level GLM."""

    analysis_type = "activation"

    def run(self, fitted: FittedModel, plan: AnalysisPlan) -> tuple[ContrastOutput | AtlasOutput | ReportOutput, ...]:
        contrasts = self._contrasts(plan.spec.configuration.get("contrasts", {}))
        output_types = self._output_types(plan.spec.configuration.get("output_types"))
        output_dir = plan.derivatives.analysis_directory
        if output_dir is None:
            raise ValueError("Activation analysis plan has no analysis derivative directory.")
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[ContrastOutput | AtlasOutput | ReportOutput] = []
        target_image = None
        contrast_effect_paths: dict[str, Path] = {}
        for name, definition in contrasts.items():
            for output_type in output_types:
                image = fitted.estimator.compute_contrast(
                    definition,
                    output_type=output_type,
                )
                target_image = image
                path = output_dir / DerivativePathBuilder.build_filename(
                    subject=fitted.plan.context.subject,
                    session=fitted.plan.context.session,
                    task=fitted.plan.context.task,
                    run=fitted.plan.context.run,
                    desc=self._component(name),
                    stat=self._stat_component(output_type),
                )
                image.to_filename(path)
                outputs.append(ContrastOutput(name, output_type, path))
                if output_type == "effect_size":
                    contrast_effect_paths[name] = path

        generate_report = getattr(fitted.estimator, "generate_report", None)
        if callable(generate_report):
            report = generate_report(
                contrasts=contrasts,
                title=(
                    f"First-level GLM: {fitted.plan.context.subject} "
                    f"{fitted.plan.context.task} {fitted.plan.context.session or ''}"
                ).strip()
            )
            report_path = output_dir / DerivativePathBuilder.build_filename(
                subject=fitted.plan.context.subject,
                session=fitted.plan.context.session,
                task=fitted.plan.context.task,
                run=fitted.plan.context.run,
                desc="first-level-report",
                suffix=".html",
            )
            report.save_as_html(str(report_path))
            metadata = {
                "model": fitted.plan.spec.name,
                "model_type": fitted.plan.spec.model_type,
                "context": fitted.plan.context.as_dict(),
                "fingerprint_configuration": fitted.plan.spec.fingerprint_configuration,
                "design_columns": list(fitted.estimator.design_matrices_[0].columns),
                "contrasts": dict(contrasts),
            }
            report_html = report_path.read_text(encoding="utf-8")
            metadata_html = (
                "<section><h2>Model metadata</h2><pre>"
                f"{html.escape(json.dumps(metadata, indent=2, default=str))}"
                "</pre></section>"
            )
            report_path.write_text(
                report_html.replace("</body>", f"{metadata_html}</body>"),
                encoding="utf-8",
            )
            outputs.append(ReportOutput(report_path))
        atlas_names = plan.spec.configuration.get("atlases", fitted.plan.spec.configuration.get("atlases", ()))
        if isinstance(atlas_names, str):
            atlas_names = [atlas_names]
        if target_image is not None and isinstance(atlas_names, (list, tuple, set)):
            cache = AtlasCache(plan.spec.configuration.get("atlas_cache_dir"))
            conditions = fitted.plan.spec.configuration.get("conditions", ())
            if isinstance(conditions, str):
                conditions = [conditions]
            condition_images = {}
            if isinstance(conditions, (list, tuple, set)):
                condition_images = {
                    str(condition): fitted.estimator.compute_contrast(str(condition), output_type="effect_size")
                    for condition in conditions
                    if str(condition).strip()
                }
            residuals = getattr(fitted.estimator, "residuals_", None)
            if residuals is None:
                residuals = getattr(fitted.estimator, "residuals", None)
            if not isinstance(residuals, (list, tuple)):
                residuals = []
            for atlas_name in atlas_names:
                atlas_text = self._component(str(atlas_name))
                path = output_dir / DerivativePathBuilder.build_filename(
                    subject=fitted.plan.context.subject,
                    session=fitted.plan.context.session,
                    task=fitted.plan.context.task,
                    run=fitted.plan.context.run,
                    desc=f"atlas-{atlas_text}-resampled",
                )
                cache.resample(str(atlas_name), target_image, path)
                outputs.append(AtlasOutput(str(atlas_name), path))
                if condition_images:
                    activation_path = output_dir / DerivativePathBuilder.build_filename(
                        subject=fitted.plan.context.subject,
                        session=fitted.plan.context.session,
                        task=fitted.plan.context.task,
                        run=fitted.plan.context.run,
                        desc=f"atlas-{atlas_text}-activation",
                        suffix=".tsv",
                    )
                    cache.roi_activation_summary(str(atlas_name), condition_images, target_image, activation_path)
                    outputs.append(AtlasOutput(f"{atlas_name}:activation", activation_path))
                if contrast_effect_paths:
                    contrast_path = output_dir / DerivativePathBuilder.build_filename(
                        subject=fitted.plan.context.subject,
                        session=fitted.plan.context.session,
                        task=fitted.plan.context.task,
                        run=fitted.plan.context.run,
                        desc=f"atlas-{atlas_text}-contrasts",
                        suffix=".tsv",
                    )
                    contrast_images = {
                        contrast: load_img(str(path))
                        for contrast, path in contrast_effect_paths.items()
                    }
                    cache.roi_contrast_summary(
                        str(atlas_name), contrast_images, target_image, contrast_path
                    )
                    outputs.append(AtlasOutput(f"{atlas_name}:contrasts", contrast_path))
                if residuals:
                    residual_path = output_dir / DerivativePathBuilder.build_filename(
                        subject=fitted.plan.context.subject,
                        session=fitted.plan.context.session,
                        task=fitted.plan.context.task,
                        run=fitted.plan.context.run,
                        desc=f"atlas-{atlas_text}-residuals",
                        suffix=".tsv",
                    )
                    cache.roi_residual_timeseries(str(atlas_name), list(residuals), target_image, residual_path)
                    outputs.append(AtlasOutput(f"{atlas_name}:residuals", residual_path))
        return tuple(outputs)

    @staticmethod
    def _contrasts(value: Any) -> Mapping[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError("Activation contrasts must be an object keyed by contrast name.")
        contrasts: dict[str, str] = {}
        for name, definition in value.items():
            contrast_name = str(name).strip()
            contrast_definition = str(definition).strip()
            if not contrast_name or not contrast_definition:
                raise ValueError("Activation contrasts require non-empty names and definitions.")
            ActivationAnalysis._component(contrast_name)
            contrasts[contrast_name] = contrast_definition
        return contrasts

    @staticmethod
    def _output_types(value: Any) -> tuple[str, ...]:
        if value is None:
            return ("effect_size", "variance", "z_score")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("Activation output_types must be a string or array.")
        output_types = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if not output_types:
            raise ValueError("Activation output_types must not be empty.")
        for output_type in output_types:
            ActivationAnalysis._component(output_type)
        return output_types

    @staticmethod
    def _component(value: str) -> str:
        text = str(value).strip()
        if not text or text in {".", ".."} or "/" in text or "\\" in text:
            raise ValueError(f"Activation output components must be single path components: {value!r}.")
        return text.replace(" ", "-")

    @staticmethod
    def _stat_component(value: str) -> str:
        return {
            "effect_size": "effect",
            "z_score": "z",
            "variance": "variance",
        }.get(value, value.replace("_", "-"))


__all__ = ["ActivationAnalysis", "AtlasOutput", "ContrastOutput", "ReportOutput"]
