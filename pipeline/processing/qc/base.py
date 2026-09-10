from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportResult:
    name: str
    status: str
    message: str
    output_dir: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)


class QCReport(ABC):
    name = "base"
    scope = "dataset"

    @abstractmethod
    def generate(self, context: Any) -> ReportResult:
        raise NotImplementedError