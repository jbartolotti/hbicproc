from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(slots=True)
class Event:
    onset: float
    duration: float
    trial_type: str
    response_time: float | None = None
    accuracy: int | None = None


@dataclass(slots=True)
class EventTable:
    records: list[Event]

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any] | Event]) -> "EventTable":
        normalized: list[Event] = []
        for record in records:
            if isinstance(record, Event):
                normalized.append(record)
                continue
            normalized.append(
                Event(
                    onset=float(record["onset"]),
                    duration=float(record["duration"]),
                    trial_type=str(record["trial_type"]),
                    response_time=float(record["response_time"]) if record.get("response_time") is not None else None,
                    accuracy=int(record["accuracy"]) if record.get("accuracy") is not None else None,
                )
            )
        return cls(records=normalized)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in self.records:
            row = {
                "onset": record.onset,
                "duration": record.duration,
                "trial_type": record.trial_type,
            }
            if record.response_time is not None:
                row["response_time"] = record.response_time
            if record.accuracy is not None:
                row["accuracy"] = record.accuracy
            rows.append(row)
        return pd.DataFrame(rows)

    def to_tsv(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(output_path, sep="\t", index=False, encoding="utf-8")
        return output_path
