from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..models.canonical_glm import FittedModel
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


class ActivationAnalysis:
    """Compute configured activation contrasts from a fitted first-level GLM."""

    analysis_type = "activation"

    def run(self, fitted: FittedModel, plan: AnalysisPlan) -> tuple[ContrastOutput, ...]:
        contrasts = self._contrasts(plan.spec.configuration.get("contrasts", {}))
        output_types = self._output_types(plan.spec.configuration.get("output_types"))
        output_dir = plan.derivatives.analysis_directory
        if output_dir is None:
            raise ValueError("Activation analysis plan has no analysis derivative directory.")
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[ContrastOutput] = []
        for name, definition in contrasts.items():
            for output_type in output_types:
                image = fitted.estimator.compute_contrast(
                    definition,
                    output_type=output_type,
                )
                path = output_dir / f"contrast-{self._component(name)}_stat-{self._component(output_type)}.nii.gz"
                image.to_filename(path)
                outputs.append(ContrastOutput(name, output_type, path))
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
            return ("effect_size", "z_score")
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


__all__ = ["ActivationAnalysis", "ContrastOutput"]
