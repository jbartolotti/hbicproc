from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ...core.subprocess_utils import run_command
from ...decisions import DecisionManifest
from .discovery import discover_effect_maps
from .masks import get_group_mask
from .participants import load_participants
from .population import select_records
from .voxelwise_lme_report import render_voxelwise_lme_report

logger = logging.getLogger(__name__)


def build_afni_data_table(records: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Build the AFNI 3dLMEr Subj/Group/Time/InputFile table."""

    joined = records.merge(participants, on="subject", how="inner", validate="many_to_one")
    if "group" not in joined.columns:
        raise ValueError("Participants must contain a 'group' column for voxelwise_lme.")
    joined["group"] = joined["group"].map(_normalize_group)
    joined["time"] = joined["session"].map(_normalize_time)
    joined = joined.dropna(subset=["subject", "group", "time", "path"]).copy()
    table = pd.DataFrame({
        "Subj": joined["subject"].astype(str),
        "Group": joined["group"].astype(str),
        "Time": joined["time"].astype(str),
        "InputFile": joined["path"].map(lambda value: str(Path(value).resolve())),
    })
    return table.sort_values(["Subj", "Time"]).reset_index(drop=True)


def build_3dlmer_command(
    *,
    output_prefix: Path,
    data_table: Path,
    mask: Path | None = None,
    jobs: int | None = None,
) -> list[str]:
    """Build the AFNI 3dLMEr command for group-by-time inference."""

    command = ["3dLMEr", "-prefix", str(output_prefix)]
    if jobs is not None:
        command.extend(["-jobs", str(int(jobs))])
    if mask is not None:
        command.extend(["-mask", str(mask)])
    command.extend([
        "-model", "Group*Time",
        "-ranEff", "~1|Subj",
        "-num_glt", "3",
        "-gltLabel", "1", "Group",
        "-gltCode", "1", "Group : 1*intervention -1*control",
        "-gltLabel", "2", "Time",
        "-gltCode", "2", "Time : 1*followup -1*baseline",
        "-gltLabel", "3", "GroupXTime",
        "-gltCode", "3", "Group : 1*intervention -1*control Time : 1*followup -1*baseline",
        "-dataTable", f"@{data_table}",
    ])
    return command


def run_voxelwise_lme(
    config: dict[str, Any],
    specification: dict[str, Any],
    *,
    decision_manifest: DecisionManifest | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare and run AFNI 3dLMEr for each configured first-level contrast."""

    analysis = config["analysis"]
    derivatives_root = Path(analysis["output_dir"])
    bids_root = Path(config.get("bids_root") or config.get("study_root", "."))
    participants = load_participants(bids_root)
    contrasts = _as_names(specification.get("contrasts", []))
    task = str(specification.get("task", "")).strip() or None
    output_root = derivatives_root / "group" / "voxelwise_lme"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for contrast in contrasts:
        records = discover_effect_maps(derivatives_root, contrast=contrast, task=task)
        if decision_manifest is not None:
            records = select_records(records, decision_manifest.population)
        if records.empty:
            raise FileNotFoundError(f"No effect maps found for configured contrast '{contrast}'.")
        table = build_afni_data_table(records, participants)
        if table.empty:
            raise ValueError(f"No usable AFNI observations found for contrast '{contrast}'.")
        contrast_dir = output_root / _component(contrast)
        contrast_dir.mkdir(parents=True, exist_ok=True)
        table_path = contrast_dir / "afni_data_table.tsv"
        table.to_csv(table_path, sep="\t", index=False)
        mask_path = _prepare_mask(config, specification, records, contrast_dir)
        command = build_3dlmer_command(
            output_prefix=contrast_dir / "3dLMEr",
            data_table=table_path,
            mask=mask_path,
            jobs=specification.get("jobs"),
        )
        version = run_command(["3dLMEr", "-ver"], dry_run=dry_run)
        started = time.perf_counter()
        execution = run_command(command, dry_run=dry_run)
        runtime = time.perf_counter() - started
        output_files = _discover_outputs(contrast_dir)
        result = {
            "analysis": "voxelwise_lme",
            "contrast": contrast,
            "n_subjects": int(table["Subj"].nunique()),
            "groups": sorted(table["Group"].unique().tolist()),
            "sessions": sorted(table["Time"].unique().tolist()),
            "data_table": str(table_path),
            "mask": str(mask_path) if mask_path else None,
            "command": execution.get("command", " ".join(command)),
            "returncode": execution.get("returncode"),
            "success": execution.get("success", False),
            "stdout": execution.get("stdout", ""),
            "stderr": execution.get("stderr", ""),
            "runtime_seconds": runtime,
            "afni_version": _version_text(version),
            "outputs": output_files,
        }
        report_path = render_voxelwise_lme_report(result, contrast_dir / "report.html")
        result["report"] = str(report_path)
        results.append(result)
        if not execution.get("success", False) and not dry_run:
            raise RuntimeError(f"AFNI 3dLMEr failed for contrast '{contrast}'.")
    return {"analysis": "voxelwise_lme", "results": results}


def _prepare_mask(config: dict[str, Any], specification: dict[str, Any], records: pd.DataFrame, output_dir: Path) -> Path | None:
    mask_config = specification.get("mask")
    mask_config = mask_config if isinstance(mask_config, dict) else config.get("group", {}).get("mask", {})
    local_config = {**config, "group": {**config.get("group", {}), "mask": mask_config}}
    reference = records.iloc[0]["path"]
    mask_image = get_group_mask(local_config, reference)
    if mask_image is None:
        return None
    mask_path = output_dir / "group_mask.nii.gz"
    mask_image.to_filename(mask_path)
    return mask_path


def _discover_outputs(output_dir: Path) -> list[dict[str, str]]:
    excluded = {"afni_data_table.tsv", "report.html", "group_mask.nii.gz"}
    outputs = []
    for path in sorted(output_dir.iterdir()):
        if path.name in excluded or not path.is_file():
            continue
        kind = "interaction" if "GroupXTime" in path.name else "main effect/output"
        if "Group" in path.name and "GroupXTime" not in path.name:
            kind = "group main effect"
        elif "Time" in path.name:
            kind = "time main effect"
        outputs.append({"kind": kind, "path": str(path)})
    return outputs


def _version_text(result: dict[str, Any]) -> str:
    text = (result.get("stdout") or result.get("stderr") or "").strip()
    return text.splitlines()[0] if text else "unknown"


def _normalize_group(value: Any) -> str | None:
    text = str(value).strip().lower()
    return text if text in {"control", "intervention"} else None


def _normalize_time(value: Any) -> str | None:
    text = str(value).strip().lower().removeprefix("ses-")
    return {"bl": "baseline", "baseline": "baseline", "w12": "followup", "followup": "followup"}.get(text)


def _as_names(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise ValueError("group.voxelwise_lme.contrasts must be a list of names.")
    return [str(value).strip() for value in values if str(value).strip()]


def _component(value: str) -> str:
    return str(value).strip().replace("_", "-").replace(" ", "-")
