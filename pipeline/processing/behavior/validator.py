from __future__ import annotations

from typing import Sequence

import pandas as pd

from .models import EventTable


REQUIRED_COLUMNS = ("onset", "duration", "trial_type")


def validate_event_table(
    events: EventTable | pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    if isinstance(events, EventTable):
        data_frame = events.to_dataframe()
    else:
        data_frame = events.copy()

    missing = [column for column in required_columns if column not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in ("onset", "duration"):
        if column in data_frame.columns:
            numeric = pd.to_numeric(data_frame[column], errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"Column '{column}' contains non-numeric values.")
            data_frame[column] = numeric

    if data_frame["onset"].lt(0).any():
        raise ValueError("'onset' must be non-negative.")
    if data_frame["duration"].lt(0).any():
        raise ValueError("'duration' must be non-negative.")

    data_frame["trial_type"] = data_frame["trial_type"].astype(str).str.strip()
    if data_frame["trial_type"].eq("").any():
        raise ValueError("'trial_type' contains empty values.")

    return data_frame
