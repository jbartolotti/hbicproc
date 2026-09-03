from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from ..context import TaskRunContext
from ..derivatives import DerivativePaths, DerivativePathBuilder


@dataclass(frozen=True)
class ModelPlan:
    """Non-executing model plan produced from a model specification."""

    spec: "ModelSpec"
    context: TaskRunContext
    derivatives: DerivativePaths

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.spec.name,
            "model_type": self.spec.model_type,
            "context": self.context.as_dict(),
            "derivatives": self.derivatives.as_dict(),
            "fingerprint_configuration": self.spec.fingerprint_configuration,
        }


@dataclass(frozen=True)
class ModelSpec:
    """Configuration-level model interface.

    Model implementations added in later PRs will subclass this object and add
    fitting and derivative-generation behavior. PR 1 intentionally only plans
    model work and never executes it.
    """

    name: str
    model_type: str
    configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Model names must not be empty.")
        if not self.model_type.strip():
            raise ValueError(f"Model '{self.name}' must declare a non-empty type.")

    @property
    def fingerprint_configuration(self) -> dict[str, Any]:
        """Return configuration that identifies the fitted GLM.

        Atlas settings intentionally do not participate in the GLM identity;
        atlas derivatives are independently regenerable model products.
        """

        configuration = deepcopy(dict(self.configuration))
        configuration.pop("atlases", None)
        return configuration

    def fingerprint_payload(self, context: TaskRunContext) -> dict[str, Any]:
        return {
            "task": context.task,
            "subject": context.subject,
            "session": context.session,
            "run": context.run,
            "model": self.name,
            "model_type": self.model_type,
            "configuration": self.fingerprint_configuration,
        }

    def plan(self, context: TaskRunContext, output_root: str) -> ModelPlan:
        derivatives = DerivativePathBuilder.for_context(
            output_root,
            context,
            namespace="model",
            name=self.name,
        )
        return ModelPlan(spec=self, context=context, derivatives=derivatives)


__all__ = ["ModelPlan", "ModelSpec"]
