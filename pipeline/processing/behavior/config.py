from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ColumnConfig:
    source: str
    name: str | None = None


@dataclass(slots=True)
class TaskConfig:
    task: str
    trigger_column: str | None = None
    event_rows: dict[str, Any] = field(default_factory=dict)
    columns: dict[str, ColumnConfig] = field(default_factory=dict)
    optional_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, ColumnConfig] = {}
        for name, spec in (self.columns or {}).items():
            if isinstance(spec, ColumnConfig):
                normalized[name] = spec
            elif isinstance(spec, dict):
                normalized[name] = ColumnConfig(
                    source=str(spec.get("source", "")),
                    name=spec.get("name"),
                )
            else:
                normalized[name] = ColumnConfig(source=str(spec or ""), name=None)
        self.columns = normalized

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Task config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        columns = {
            name: ColumnConfig(source=spec.get("source", ""), name=spec.get("name"))
            for name, spec in (raw.get("columns") or {}).items()
        }
        return cls(
            task=str(raw.get("task", path.stem)),
            trigger_column=raw.get("trigger_column"),
            event_rows=raw.get("event_rows") or {},
            columns=columns,
            optional_columns=list(raw.get("optional_columns") or []),
            metadata=raw.get("metadata") or raw.get("sidecar") or {},
        )

    def column_sources(self) -> dict[str, str]:
        return {name: spec.source for name, spec in self.columns.items() if spec.source}
