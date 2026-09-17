from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def select_records(records: pd.DataFrame, population: Mapping[str, Any]) -> pd.DataFrame:
    """Apply investigator population decisions to discovered group records."""

    if records.empty:
        return records
    selected = records.copy()
    for column in ("subject", "session", "run"):
        if column not in selected.columns:
            selected[column] = None
        selected[column] = selected[column].fillna("n/a").astype(str).map(_entity_value)

    include_rules = population.get("include", [])
    exclude_rules = population.get("exclude", [])
    if include_rules:
        selected = selected[
            selected.apply(lambda row: _matches_any(row, include_rules), axis=1)
        ].copy()
    if exclude_rules:
        selected = selected[
            ~selected.apply(lambda row: _matches_any(row, exclude_rules), axis=1)
        ].copy()
    return selected


def _matches_any(row: pd.Series, rules: Any) -> bool:
    return any(_matches_rule(row, rule) for rule in rules)


def _matches_rule(row: pd.Series, rule: Mapping[str, Any]) -> bool:
    if _entity_value(rule.get("subject", "")) != row["subject"]:
        return False
    for field_name in ("session", "run"):
        if field_name in rule and _entity_value(rule[field_name]) != row[field_name]:
            return False
    return True


def _entity_value(value: Any) -> str:
    text = str(value).strip()
    for prefix in ("sub-", "ses-", "run-"):
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text
