from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputDataset:
    """Named preprocessed input dataset used by an analysis task."""

    name: str
    path: Path

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "path": str(self.path)}


@dataclass(frozen=True)
class TaskRunContext:
    """Stable identity and input paths for one task/session/run.

    The context deliberately contains no model or analysis state. This keeps the
    same object usable by model implementations, analyses, and future run-group
    planners that may combine multiple contexts.
    """

    subject: str
    task: str
    session: str | None = None
    run: str | None = None
    bold_path: Path | None = None
    events_path: Path | None = None
    confounds_path: Path | None = None
    input_dataset: InputDataset | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "task": self.task,
            "session": self.session,
            "run": self.run,
            "bold_path": str(self.bold_path) if self.bold_path is not None else None,
            "events_path": str(self.events_path) if self.events_path is not None else None,
            "confounds_path": str(self.confounds_path) if self.confounds_path is not None else None,
            "input_dataset": self.input_dataset.as_dict() if self.input_dataset else None,
        }

    @property
    def identity(self) -> tuple[str, str, str | None, str | None]:
        return self.subject, self.task, self.session, self.run


__all__ = ["InputDataset", "TaskRunContext"]
