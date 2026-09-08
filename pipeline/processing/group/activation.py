from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
from nilearn.glm.second_level import SecondLevelModel

from .discovery import discover_effect_maps
from .participants import add_factor_columns, load_participants


def _component(value: str) -> str:
    return str(value).strip().replace("_", "-").replace(" ", "-")


def _levels(values: pd.Series, configured: Any = None) -> list[str]:
    if isinstance(configured, (list, tuple)):
        return [str(value) for value in configured]
    return sorted(str(value) for value in values.dropna().unique())


def _activation_design(records: pd.DataFrame, factors: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if "group" not in factors or "session" not in factors:
        raise ValueError("Group activation requires configured group and session factors.")
    group_levels = _levels(records["group"], factors["group"].get("levels"))
    session_levels = _levels(records["session"], factors["session"].get("levels"))
    if len(group_levels) != 2 or len(session_levels) != 2:
        raise ValueError("Group activation requires exactly two group and two session levels.")
    group_indicator = (records["group"].astype(str) == group_levels[1]).astype(int)
    session_indicator = (records["session"].astype(str) == session_levels[1]).astype(int)
    design = pd.DataFrame({
        "intercept": 1.0,
        "group": group_indicator,
        "session": session_indicator,
        "group_session": group_indicator * session_indicator,
    })
    return design, {"group": group_levels, "session": session_levels}


def run_activation(config: dict[str, Any], specification: dict[str, Any]) -> dict[str, Any]:
    analysis = config["analysis"]
    derivatives_root = Path(analysis["output_dir"])
    bids_root = Path(config.get("bids_root") or config.get("study_root", "."))
    task = str(specification.get("task", "")).strip()
    factors = specification.get("factors", {})
    contrasts = specification.get("contrasts", [])
    if not task or not isinstance(factors, dict) or not isinstance(contrasts, (list, tuple)):
        raise ValueError("Group activation requires task, factors, and contrasts configuration.")

    participants = load_participants(bids_root)
    results = []
    for contrast in contrasts:
        contrast_name = str(contrast).strip()
        records = discover_effect_maps(derivatives_root, task=task, contrast=contrast_name)
        if records.empty:
            raise FileNotFoundError(f"No effect maps found for configured contrast '{contrast_name}'.")
        records = add_factor_columns(records, participants, factors)
        records = records.dropna(subset=["group", "session"])
        design, levels = _activation_design(records, factors)
        if len(records) < 4:
            raise ValueError(f"Insufficient observations for group contrast '{contrast_name}'.")

        model = SecondLevelModel(smoothing_fwhm=None, minimize_memory=False)
        model.fit(records["path"].tolist(), design_matrix=design)
        output_dir = derivatives_root / "group" / "activation" / _component(contrast_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        design_path = output_dir / "design_matrix.tsv"
        design.to_csv(design_path, sep="\t", index=False)
        vectors = {
            "interaction": [0, 0, 0, 1],
            "session": [0, 0, 1, 0],
            "group": [0, 1, 0, 0],
        }
        output_paths = {"design_matrix": str(design_path)}
        for name, vector in vectors.items():
            effect = model.compute_contrast(vector, output_type="effect_size")
            z_score = model.compute_contrast(vector, output_type="z_score")
            effect_path = output_dir / f"group_desc-{name}_stat-effect.nii.gz"
            z_path = output_dir / f"group_desc-{name}_stat-z.nii.gz"
            effect.to_filename(effect_path)
            z_score.to_filename(z_path)
            output_paths[f"{name}_effect"] = str(effect_path)
            output_paths[f"{name}_z"] = str(z_path)

        report_path = output_dir / "report.html"
        _write_activation_report(report_path, contrast_name, records, design, levels)
        output_paths["report"] = str(report_path)
        results.append({"contrast": contrast_name, "n_maps": len(records), "outputs": output_paths})
    return {"analysis": "activation", "results": results}


def _write_activation_report(path: Path, contrast: str, records: pd.DataFrame, design: pd.DataFrame, levels: dict[str, list[str]]) -> None:
    metadata = {
        "contrast": contrast,
        "n_maps": len(records),
        "group_levels": levels["group"],
        "session_levels": levels["session"],
    }
    path.write_text(
        "<html><body><h1>Group activation: "
        f"{html.escape(contrast)}</h1><h2>Metadata</h2><pre>"
        f"{html.escape(json.dumps(metadata, indent=2))}</pre><h2>Design matrix</h2>"
        f"{design.to_html(index=False)}</body></html>",
        encoding="utf-8",
    )
