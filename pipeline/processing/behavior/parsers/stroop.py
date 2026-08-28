from __future__ import annotations

import pandas as pd

from ..config import TaskConfig
from .base import TaskParser


class StroopParser(TaskParser):
    def parse(self, raw_df: pd.DataFrame, config: TaskConfig) -> pd.DataFrame:
        filtered = raw_df.copy()
        event_rows = config.event_rows or {}
        filter_column = event_rows.get("filter_column")
        include_values = event_rows.get("include_values") or []

        if filter_column and include_values:
            values = {str(value).strip().lower() for value in include_values}
            filtered = filtered[filtered[filter_column].astype(str).str.strip().str.lower().isin(values)].copy()

        if config.trigger_column and config.trigger_column in filtered.columns:
            trigger_time = pd.to_numeric(filtered[config.trigger_column], errors="coerce")
            trigger_time = trigger_time.fillna(0.0)
        else:
            trigger_time = pd.Series(0.0, index=filtered.index)

        onset_source = config.columns.get("onset", None)
        duration_source = config.columns.get("duration", None)
        trial_type_source = config.columns.get("trial_type", None)

        onset_col = onset_source.source if onset_source else None
        duration_col = duration_source.source if duration_source else None
        trial_type_col = trial_type_source.source if trial_type_source else None

        if not onset_col or onset_col not in filtered.columns:
            raise ValueError(f"Missing onset column '{onset_col}'.")
        if not duration_col or duration_col not in filtered.columns:
            raise ValueError(f"Missing duration column '{duration_col}'.")
        if not trial_type_col or trial_type_col not in filtered.columns:
            raise ValueError(f"Missing trial_type column '{trial_type_col}'.")

        events = pd.DataFrame(
            {
                "onset": pd.to_numeric(filtered[onset_col], errors="coerce") - trigger_time,
                "duration": pd.to_numeric(filtered[duration_col], errors="coerce"),
                "trial_type": filtered[trial_type_col].astype(str).str.strip().str.lower(),
            }
        )

        for column_name in config.optional_columns:
            source_name = column_name
            if column_name in filtered.columns:
                source_name = column_name
            elif column_name.lower() in filtered.columns:
                source_name = column_name.lower()
            else:
                continue
            events[column_name.lower()] = pd.to_numeric(filtered[source_name], errors="coerce")

        events = events.reset_index(drop=True)
        return events
