from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_participants(bids_root: str | Path) -> pd.DataFrame:
    """Load authoritative participant metadata from the BIDS root."""

    path = Path(bids_root) / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Participants file not found: {path}")
    participants = pd.read_csv(path, sep="\t")
    if "participant_id" not in participants.columns:
        raise ValueError(f"Participants file {path} must contain participant_id.")
    participants = participants.copy()
    participants["subject"] = participants["participant_id"].astype(str).str.replace(
        r"^(sub[-_])", "", regex=True
    )
    return participants


def add_factor_columns(
    records: pd.DataFrame,
    participants: pd.DataFrame,
    factors: dict[str, Any],
) -> pd.DataFrame:
    """Merge configured participant and BIDS-session factors into derivative records."""

    result = records.merge(participants, on="subject", how="left", validate="many_to_one")
    for factor_name, factor in factors.items():
        if not isinstance(factor, dict):
            raise ValueError(f"Group factor '{factor_name}' must be an object.")
        source = factor.get("source")
        if source == "participants.tsv":
            column = str(factor.get("column", "")).strip()
            if not column or column not in result.columns:
                raise ValueError(f"Participant factor '{factor_name}' requires a valid column.")
            result[factor_name] = result[column]
        elif source == "bids_session":
            result[factor_name] = result["session"]
            levels = factor.get("levels", {})
            if isinstance(levels, dict):
                result[factor_name] = result[factor_name].map(
                    {str(key).removeprefix("ses-"): value for key, value in levels.items()}
                )
        else:
            raise ValueError(
                f"Unsupported source for group factor '{factor_name}': {source!r}."
            )
    return result
