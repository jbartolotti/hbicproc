from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..context import TaskRunContext
from ..derivatives import DerivativePaths, DerivativePathBuilder
from ..models.base import ModelSpec


@dataclass(frozen=True)
class AnalysisPlan:
    """Non-executing analysis plan bound to a model specification."""

    spec: "AnalysisSpec"
    context: TaskRunContext
    model: ModelSpec
    derivatives: DerivativePaths

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.spec.name,
            "model": self.model.name,
            "context": self.context.as_dict(),
            "derivatives": self.derivatives.as_dict(),
            "required_derivatives": list(self.spec.required_derivatives),
        }


@dataclass(frozen=True)
class AnalysisSpec:
    """Configuration-level analysis interface.

    Concrete analyses added in later PRs consume model products and write
    analysis-specific products. This interface does not fit models or produce
    scientific outputs by itself.
    """

    name: str
    model: str
    configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Analysis names must not be empty.")
        if not self.model.strip():
            raise ValueError(f"Analysis '{self.name}' must reference a model.")

    @property
    def required_derivatives(self) -> tuple[str, ...]:
        requested = self.configuration.get("requires", ())
        if isinstance(requested, str):
            requested = (requested,)
        if not isinstance(requested, (list, tuple, set)):
            return ()
        return tuple(dict.fromkeys(str(item).strip() for item in requested if str(item).strip()))

    def plan(
        self,
        context: TaskRunContext,
        model: ModelSpec,
        output_root: str,
    ) -> AnalysisPlan:
        derivatives = DerivativePathBuilder.for_context(
            output_root,
            context,
            namespace="analysis",
            name=self.name,
        )
        return AnalysisPlan(
            spec=self,
            context=context,
            model=model,
            derivatives=derivatives,
        )


__all__ = ["AnalysisPlan", "AnalysisSpec"]
