from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .atlas_metadata import AtlasMetadata
from .discovery import discover_atlas_activation_summaries
from .participants import add_factor_columns, load_participants
from ..analysis.analyses.atlas import AtlasCache


def run_dmn(config: dict, specification: dict) -> dict:
    analysis = config["analysis"]
    derivatives_root = Path(analysis["output_dir"])
    bids_root = Path(config.get("bids_root") or config.get("study_root", "."))
    task = str(specification.get("task", "")).strip()
    atlas_name = str(specification.get("atlas", "")).strip()
    network = str(specification.get("network", "Default")).strip()
    factors = specification.get("factors", {})
    contrasts = specification.get("contrasts", [])
    if not task or not atlas_name or not isinstance(factors, dict):
        raise ValueError("DMN analysis requires task, atlas, and factors configuration.")

    participants = load_participants(bids_root)
    atlas_cache = AtlasCache(specification.get("atlas_cache_dir"))
    atlas_metadata = AtlasMetadata.from_cache(atlas_cache, atlas_name)
    parcel_ids = {parcel.parcel_id for parcel in atlas_metadata.get_parcels(network=network)}
    if not parcel_ids:
        raise ValueError(f"No parcels found for network '{network}' in atlas '{atlas_name}'.")

    rows = []
    for contrast in contrasts:
        contrast_name = str(contrast).strip()
        summaries = discover_atlas_activation_summaries(
            derivatives_root, task=task, atlas=atlas_name
        )
        if summaries.empty:
            raise FileNotFoundError(f"No activation summaries found for atlas '{atlas_name}'.")
        summaries["roi"] = pd.to_numeric(summaries["roi"], errors="coerce")
        selected = summaries[summaries["roi"].isin(parcel_ids)].copy()
        selected["dmn_value"] = _contrast_values(selected, contrast_name)
        selected = selected.dropna(subset=["dmn_value"])
        subject_values = selected.groupby(["subject", "session"], dropna=False)["dmn_value"].mean().reset_index()
        subject_values["contrast"] = contrast_name
        rows.append(add_factor_columns(subject_values, participants, factors)[
            ["subject", "session", "group", "contrast", "dmn_value"]
        ])

    values = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output_root = derivatives_root / "group" / "dmn"
    output_root.mkdir(parents=True, exist_ok=True)
    values_path = output_root / "dmn_values.tsv"
    values.to_csv(values_path, sep="\t", index=False)
    stats = _fit_stats(values)
    stats_path = output_root / "dmn_stats.tsv"
    stats.to_csv(stats_path, sep="\t", index=False)
    report_path = output_root / "dmn_plots.html"
    report_path.write_text(
        "<html><body><h1>DMN group analysis</h1>"
        f"<p>Atlas: {html.escape(atlas_name)}; network: {html.escape(network)}</p>"
        f"{stats.to_html(index=False)}</body></html>",
        encoding="utf-8",
    )
    return {
        "analysis": "dmn",
        "values": str(values_path),
        "stats": str(stats_path),
        "report": str(report_path),
    }


def _contrast_values(summary: pd.DataFrame, contrast: str) -> pd.Series:
    """Resolve a contrast column or derive a simple configured ``A_gt_B`` contrast."""

    candidates = (contrast, contrast.replace("_", "-"))
    for candidate in candidates:
        if candidate in summary.columns:
            return pd.to_numeric(summary[candidate], errors="coerce")
    parts = contrast.split("_gt_", maxsplit=1)
    if len(parts) == 2 and all(part in summary.columns for part in parts):
        return pd.to_numeric(summary[parts[0]], errors="coerce") - pd.to_numeric(
            summary[parts[1]], errors="coerce"
        )
    raise ValueError(
        f"Activation summaries do not contain contrast '{contrast}' or both condition columns "
        f"needed to derive it."
    )


def _fit_stats(values: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    rows = []
    for contrast, frame in values.groupby("contrast"):
        if frame["group"].nunique() < 2 or frame["session"].nunique() < 2:
            continue
        fitted = smf.ols("dmn_value ~ C(group) * C(session)", data=frame).fit()
        for term in fitted.params.index:
            rows.append({
                "contrast": contrast,
                "term": term,
                "estimate": fitted.params[term],
                "t": fitted.tvalues[term],
                "p_value": fitted.pvalues[term],
            })
    return pd.DataFrame(rows, columns=["contrast", "term", "estimate", "t", "p_value"])
