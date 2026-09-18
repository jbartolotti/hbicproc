from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ...decisions import DecisionManifest
from ..analysis.analyses.atlas import AtlasCache
from .atlas_metadata import AtlasMetadata
from .discovery import discover_atlas_contrast_summaries
from .participants import add_factor_columns, load_participants
from .population import select_records
from .roi_lmm_model import apply_interaction_fdr, fit_network_lmm
from .roi_lmm_report import render_roi_lmm_report

logger = logging.getLogger(__name__)


def run_roi_lmm(
    config: dict[str, Any],
    specification: dict[str, Any],
    *,
    decision_manifest: DecisionManifest | None = None,
) -> dict[str, Any]:
    """Run longitudinal mixed-effects models over network or parcel effects."""

    analysis = config["analysis"]
    derivatives_root = Path(analysis["output_dir"])
    bids_root = Path(config.get("bids_root") or config.get("study_root", "."))
    task = str(specification.get("task", "")).strip()
    atlas_name = str(specification.get("atlas", "")).strip()
    contrasts = [str(value).strip() for value in specification.get("contrasts", [])]
    factors = specification.get("factors", {})
    if not task or not atlas_name or not contrasts or not isinstance(factors, dict):
        raise ValueError("ROI LMM requires task, atlas, contrasts, and factors configuration.")

    participants = load_participants(bids_root)
    atlas_cache = AtlasCache(specification.get("atlas_cache_dir"))
    atlas_metadata = AtlasMetadata.from_cache(atlas_cache, atlas_name)
    parcel_networks = {parcel.parcel_id: parcel.network for parcel in atlas_metadata.parcels}
    output_root = derivatives_root / "group" / "roi_lmm"
    output_root.mkdir(parents=True, exist_ok=True)
    contrast_results = []
    for contrast in contrasts:
        summaries = discover_atlas_contrast_summaries(
            derivatives_root, task=task, atlas=atlas_name, contrast=contrast
        )
        if summaries.empty:
            raise FileNotFoundError(
                f"No atlas contrast summaries found for task '{task}', atlas '{atlas_name}', contrast '{contrast}'."
            )
        aggregation = str(specification.get("aggregation", "networks")).strip().lower()
        values = _roi_values(
            summaries,
            parcel_networks,
            aggregation=aggregation,
            network_filter=specification.get("network_filter", []),
        )
        values = add_factor_columns(values, participants, factors)
        values["group"] = values["group"].astype(str).str.strip().str.lower()
        values["time"] = values["time"].astype(str).str.strip().str.lower()
        values = values.dropna(subset=["group", "time", "effect"])
        if decision_manifest is not None:
            values = select_records(values, decision_manifest.population)
        if values.empty:
            raise ValueError(f"No ROI LMM observations remain for contrast '{contrast}'.")
        unit_results = []
        for unit, frame in values.groupby("unit", sort=True):
            try:
                fitted = fit_network_lmm(
                    frame[["subject", "group", "time", "effect"]],
                    random_slope_time=bool(specification.get("random_slope_time", True)),
                    network_name=str(unit),
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning("Skipping ROI unit '%s' for contrast '%s': %s", unit, contrast, exc)
                continue
            fitted["unit"] = unit
            if aggregation == "networks":
                fitted["network"] = unit
            else:
                parcel = frame.iloc[0]
                fitted["network"] = parcel["network"]
                fitted["parcel_id"] = int(parcel["parcel_id"])
                fitted["parcel_label"] = parcel["parcel_label"]
            unit_results.append(fitted)
        if bool(specification.get("fdr_correction", True)):
            apply_interaction_fdr(unit_results, float(specification.get("alpha", 0.05)))
        contrast_dir = output_root / contrast.replace("/", "-")
        contrast_dir.mkdir(parents=True, exist_ok=True)
        values_path = contrast_dir / (
            "network_values.tsv" if aggregation == "networks" else "roi_values.tsv"
        )
        values.to_csv(values_path, sep="\t", index=False)
        plot_paths = {}
        for unit in unit_results:
            unit_path = contrast_dir / f"{_filename_component(unit['unit'])}_interaction_plot.png"
            plot_path = _interaction_plot(unit, unit_path)
            unit["plot"] = str(plot_path)
            plot_paths[unit["unit"]] = str(plot_path)
        result = {
            "analysis": "roi_lmm",
            "contrast": contrast,
            "atlas": atlas_name,
            "aggregation": aggregation,
            "network_filter": specification.get("network_filter", []),
            "n_subjects": int(values["subject"].nunique()),
            "n_units": len(unit_results),
            "n_networks": len(unit_results) if aggregation == "networks" else None,
            "model_formula": "effect ~ group_code * time_code",
            "group_coding": {"control": -0.5, "intervention": 0.5},
            "time_coding": {"baseline": -0.5, "followup": 0.5},
            "units": unit_results,
            "networks": unit_results,
            "values": str(values_path),
            "plots": plot_paths,
        }
        report_path = render_roi_lmm_report(result, contrast_dir / "report.html")
        result["report"] = str(report_path)
        contrast_results.append(result)
    return {"analysis": "roi_lmm", "results": contrast_results}


def _roi_values(
    summaries: pd.DataFrame,
    parcel_networks: dict[int, str],
    *,
    aggregation: str = "networks",
    network_filter: str | list[str] | None = None,
) -> pd.DataFrame:
    """Prepare network-aggregated or individual-parcel longitudinal values."""

    if aggregation not in {"networks", "individual"}:
        raise ValueError("ROI LMM aggregation must be 'networks' or 'individual'.")
    values = summaries.copy()
    values["parcel_id"] = pd.to_numeric(values["parcel_id"], errors="coerce")
    values["effect"] = pd.to_numeric(values["effect"], errors="coerce")
    values["parcel_label"] = values["parcel_label"].astype(str).str.strip()
    values["network"] = values["network"].astype(str).str.strip()
    missing_network = values["network"].isin({"", "nan", "None"})
    values.loc[missing_network, "network"] = values.loc[missing_network, "parcel_id"].map(parcel_networks)
    values = values.dropna(subset=["subject", "network", "effect"])
    if network_filter:
        selected = [network_filter] if isinstance(network_filter, str) else network_filter
        selected_keys = {str(value).strip().lower().replace(" ", "") for value in selected}
        values = values.loc[
            values["network"].map(
                lambda value: str(value).lower().replace(" ", "")
            ).isin(selected_keys)
        ]
    if aggregation == "networks":
        values = values.groupby(
            ["subject", "session", "network"], dropna=False, as_index=False
        )["effect"].mean()
        values["unit"] = values["network"]
        return values
    return values.groupby(
        ["subject", "session", "parcel_id", "parcel_label", "network"],
        dropna=False,
        as_index=False,
    )["effect"].mean().assign(unit=lambda frame: frame["parcel_label"])


def _network_values(summaries: pd.DataFrame, parcel_networks: dict[int, str]) -> pd.DataFrame:
    """Backward-compatible wrapper for network aggregation callers."""

    return _roi_values(summaries, parcel_networks)


def _interaction_plot(network: dict[str, Any], path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    means = {(row["group"], row["time"]): row for row in network["emmeans"]}
    for group, color in (("control", "0.35"), ("intervention", "C0")):
        baseline = means[(group, "baseline")]
        followup = means[(group, "followup")]
        axis.errorbar(
            [0, 1],
            [baseline["estimate"], followup["estimate"]],
            yerr=[
                [baseline["estimate"] - baseline["lower_ci"], followup["estimate"] - followup["lower_ci"]],
                [baseline["upper_ci"] - baseline["estimate"], followup["upper_ci"] - followup["estimate"]],
            ],
            color=color,
            marker="o",
            capsize=4,
            label=group.title(),
        )
    axis.set_xticks([0, 1], ["Baseline", "Followup"])
    axis.set_ylabel("Mean ROI effect")
    axis.set_title(f"{network['network']} estimated marginal means")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _filename_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "-" for character in value.lower())
