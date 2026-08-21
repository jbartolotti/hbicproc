from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class EventsWriter:
    def write(
        self,
        events_df: pd.DataFrame,
        output_path: str | Path,
        *,
        sidecar_path: str | Path | None = None,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path | None]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        normalized = events_df.copy().reset_index(drop=True)
        normalized.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

        sidecar_output: Path | None = None
        if sidecar_path is not None:
            sidecar_output = Path(sidecar_path)
            sidecar_output.parent.mkdir(parents=True, exist_ok=True)
            sidecar_payload = metadata or self._build_default_sidecar(normalized, task_name=task_name)
            with sidecar_output.open("w", encoding="utf-8") as handle:
                json.dump(sidecar_payload, handle, indent=2)
                handle.write("\n")

        return output_path, sidecar_output

    def _build_default_sidecar(self, events_df: pd.DataFrame, task_name: str | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if task_name:
            metadata["TaskName"] = task_name

        for column in events_df.columns:
            if column == "onset":
                description = "Event onset time in seconds relative to the start of the scan."
            elif column == "duration":
                description = "Event duration in seconds."
            elif column == "trial_type":
                description = "Label of the experimental condition or trial type."
            elif column.lower() == "accuracy":
                description = "Trial accuracy coded as 1 for correct and 0 for incorrect."
            elif column.lower() in {"rt", "response_time"}:
                description = "Reaction time in seconds."
            else:
                description = f"Behavioral column {column}."
            metadata[column] = {"Description": description}
        return {"metadata": metadata}
