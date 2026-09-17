from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, view_img

from .discovery import discover_effect_maps
from .inference import InferenceResult, create_inference_method
from .participants import load_participants
from .population import select_records
from ...decisions import DecisionManifest

logger = logging.getLogger(__name__)


def _component(value: str) -> str:
    text = str(value).strip().replace("_", "-").replace(" ", "-")
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Group report components must be single path components: {value!r}.")
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text)


def _session_value(value: Any) -> str:
    text = str(value).strip()
    return text[4:] if text.lower().startswith("ses-") else text


def _configured_sessions(value: Any) -> list[str]:
    if value is None or value == []:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("group.one_sample.sessions must be a string or list of names.")
    return list(dict.fromkeys(_session_value(item) for item in values if str(item).strip()))


def _select_records(
    records: pd.DataFrame,
    participants: pd.DataFrame,
    specification: dict[str, Any],
    decision_manifest: DecisionManifest | None = None,
) -> pd.DataFrame:
    if records.empty:
        return records
    selected = records.merge(participants[["subject"]], on="subject", how="inner", validate="many_to_one")
    if decision_manifest is not None:
        before_decisions = selected.copy()
        selected = select_records(selected, decision_manifest.population)
        removed = before_decisions.loc[~before_decisions["path"].isin(selected["path"])]
        print(
            f"[group] decision population for '{decision_manifest.analysis_id}': "
            f"{len(selected)}/{len(before_decisions)} maps retained"
        )
        for record in removed.itertuples(index=False):
            print(
                "[group] excluded map: "
                f"subject={record.subject} session={record.session or 'n/a'} "
                f"run={getattr(record, 'run', None) or 'n/a'} path={record.path}"
            )
    selected["session"] = selected["session"].fillna("n/a").astype(str)
    selected["task"] = selected["task"].fillna("n/a").astype(str)
    sessions = _configured_sessions(specification.get("sessions"))
    if sessions:
        selected = selected[selected["session"].map(_session_value).isin(sessions)].copy()
    task = str(specification.get("task", "")).strip()
    if task:
        selected = selected[selected["task"] == task.replace("task-", "")].copy()
    elif selected["task"].nunique() > 1:
        tasks = ", ".join(sorted(selected["task"].unique()))
        raise ValueError(
            "One-sample group discovery found multiple tasks for the requested contrast "
            f"({tasks}); configure group.one_sample.task."
        )
    duplicates = selected[selected.duplicated(["subject", "session", "task"], keep=False)]
    if not duplicates.empty:
        subjects = ", ".join(sorted(duplicates["subject"].astype(str).unique()))
        raise ValueError(
            "One-sample group analysis requires one map per subject/session/task; "
            f"duplicate maps were found for: {subjects}."
        )
    return selected.sort_values(["session", "subject", "task"]).reset_index(drop=True)


def _z_coordinates(image: nib.Nifti1Image, mask: np.ndarray, count: int = 10) -> np.ndarray:
    data = mask if np.any(mask) else np.isfinite(image.get_fdata())
    indices = np.argwhere(data)
    if not len(indices):
        return np.array([0.0])
    world = nib.affines.apply_affine(image.affine, indices)
    minimum, maximum = float(world[:, 2].min()), float(world[:, 2].max())
    return np.unique(np.linspace(minimum, maximum, min(count, max(1, len(np.unique(world[:, 2]))))))


def _placeholder_figure(path: Path, title: str, message: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 2.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _montages(
    inference: InferenceResult, thresholded_figure: Path, unthresholded_figure: Path, title: str
) -> None:
    image = nib.load(str(inference.stat_map))
    mask = np.asarray(nib.load(str(inference.significance_mask)).get_fdata(), dtype=bool)
    cut_coords = _z_coordinates(image, mask)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(max(8, len(cut_coords) * 1.2), 2.5))
        display = plot_stat_map(
            str(inference.thresholded_map), display_mode="z", cut_coords=cut_coords,
            threshold=0, colorbar=True, cmap="RdBu_r", symmetric_cbar=True,
            radiological=False, figure=figure, title=title + " thresholded",
        )
        figure.savefig(thresholded_figure, dpi=180, bbox_inches="tight")
        plt.close(figure)

        figure = plt.figure(figsize=(max(8, len(cut_coords) * 1.2), 2.5))
        display = plot_stat_map(
            str(inference.stat_map), display_mode="z", cut_coords=cut_coords,
            threshold=0, colorbar=True, cmap="RdBu_r", symmetric_cbar=True,
            radiological=False, figure=figure, title=title + " unthresholded",
        )
        display.add_contours(str(inference.significance_mask), levels=[0.5], colors="black", linewidths=1.2)
        figure.savefig(unthresholded_figure, dpi=180, bbox_inches="tight")
        plt.close(figure)
    except Exception:
        logger.exception("Could not generate static montages for %s", title)
        _placeholder_figure(thresholded_figure, title, "Static thresholded montage unavailable.")
        _placeholder_figure(unthresholded_figure, title, "Static unthresholded montage unavailable.")


def _html_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "<p>No clusters survived the configured threshold.</p>"
    return table.to_html(index=False, border=0, classes="data-table", escape=True, float_format=lambda value: f"{value:.3f}")


def _write_report(path: Path, sections: list[dict[str, Any]]) -> None:
    section_html = []
    for section in sections:
        viewer = (
            f"<iframe class='viewer' src='{html.escape(section['viewer'])}' title='Interactive group map'></iframe>"
            if section.get("viewer") else "<p>Interactive viewer unavailable.</p>"
        )
        section_html.append(
            f"<section><h2>Contrast: {html.escape(section['contrast'])} | Session: {html.escape(section['session'])}</h2>"
            "<h3>Summary</h3><ul>"
            f"<li>Included subjects: {section['n_subjects']}</li>"
            f"<li>Group model: one-sample intercept-only</li>"
            f"<li>Inference Method: {html.escape(section['inference']['method_label'])}</li>"
            f"<li>Alpha: {section['inference']['alpha']}</li>"
            f"<li>Computed Threshold: Z = {section['inference']['computed_threshold']:.3f}</li></ul>"
            "<h3>Thresholded group map</h3>"
            f"<img class='montage' src='{html.escape(section['thresholded_figure'])}' alt='Thresholded group map'>"
            "<h3>Unthresholded group map with cluster outlines</h3>"
            f"<img class='montage' src='{html.escape(section['unthresholded_figure'])}' alt='Unthresholded group map'>"
            "<h3>Interactive viewer</h3>" + viewer +
            "<h3>Cluster table</h3>" + section["cluster_table"] + "</section>"
        )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>One-Sample Group Analysis</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;color:#222}section{border-top:2px solid #ccc;margin-top:2rem;padding-top:1rem}.montage{display:block;max-width:100%;height:auto}.data-table{border-collapse:collapse;width:100%;margin:1rem 0}.data-table th,.data-table td{border:1px solid #ccc;padding:.35rem;text-align:left}.data-table th{background:#eee}.viewer{width:100%;height:650px;border:1px solid #bbb}</style></head><body>"
        "<h1>One-Sample Group Analysis</h1>" + "".join(section_html) + "</body></html>\n",
        encoding="utf-8",
    )


def run_one_sample(
    config: dict[str, Any],
    specification: dict[str, Any],
    *,
    decision_manifest: DecisionManifest | None = None,
) -> dict[str, Any]:
    analysis_root = Path(config["analysis"]["output_dir"])
    bids_root = Path(config.get("bids_root") or config.get("study_root", "."))
    participants = load_participants(bids_root)
    contrasts = specification.get("contrasts", [])
    inference_configuration = specification.get("inference", {})
    inference_method = create_inference_method(inference_configuration)
    report_root = analysis_root / "group" / "one_sample"
    report_root.mkdir(parents=True, exist_ok=True)
    sections = []
    results = []
    for configured_contrast in contrasts:
        contrast = str(configured_contrast).strip()
        records = discover_effect_maps(analysis_root, contrast=contrast, task=specification.get("task"))
        if records.empty:
            raise FileNotFoundError(f"No effect maps found for configured contrast '{contrast}'.")
        records = _select_records(records, participants, specification, decision_manifest)
        if records.empty:
            raise FileNotFoundError(f"No participant maps remain for contrast '{contrast}' and configured sessions.")
        for (session, task), session_records in records.groupby(["session", "task"], sort=True):
            output_dir = report_root / _component(contrast) / _component(session)
            output_dir.mkdir(parents=True, exist_ok=True)
            design = pd.DataFrame({"intercept": 1.0}, index=session_records.index)
            design.to_csv(output_dir / "design_matrix.tsv", sep="\t", index=False)
            logger.info("Fitting one-sample group model: contrast=%s session=%s subjects=%d", contrast, session, session_records["subject"].nunique())
            model = SecondLevelModel(smoothing_fwhm=None, minimize_memory=False)
            model.fit(session_records["path"].tolist(), design_matrix=design)
            effect = model.compute_contrast([1.0], output_type="effect_size")
            z_score = model.compute_contrast([1.0], output_type="z_score")
            effect_path = output_dir / "group_stat-effect.nii.gz"
            z_path = output_dir / "group_stat-z.nii.gz"
            effect.to_filename(effect_path)
            z_score.to_filename(z_path)
            inference = inference_method.run(z_path, output_dir / "inference", inference_configuration)
            clusters = pd.read_csv(inference.cluster_table, sep="\t")
            thresholded_figure = output_dir / "thresholded_map.png"
            unthresholded_figure = output_dir / "unthresholded_map.png"
            label = f"{contrast} / session {session}"
            _montages(inference, thresholded_figure, unthresholded_figure, label)
            viewer_path = output_dir / "viewer.html"
            try:
                view_img(str(inference.stat_map), title=label, threshold=inference.metadata["computed_threshold"], cmap="RdBu_r", symmetric_cmap=True).save_as_html(str(viewer_path))
                viewer = viewer_path.relative_to(analysis_root / "group").as_posix()
            except Exception:
                logger.exception("Could not generate interactive viewer for %s", label)
                viewer = None
            section = {
                "contrast": contrast,
                "session": session,
                "task": task,
                "n_subjects": int(session_records["subject"].nunique()),
                "inference": inference.metadata,
                "thresholded_figure": thresholded_figure.relative_to(analysis_root / "group").as_posix(),
                "unthresholded_figure": unthresholded_figure.relative_to(analysis_root / "group").as_posix(),
                "viewer": viewer,
                "cluster_table": _html_table(clusters),
            }
            sections.append(section)
            results.append({"contrast": contrast, "session": session, "task": task, "n_subjects": section["n_subjects"], "output_dir": str(output_dir), "n_clusters": len(clusters), "inference": inference.metadata})
    report_path = analysis_root / "group" / "one_sample_group_report.html"
    _write_report(report_path, sections)
    metadata = {
        "report_type": "one_sample_group",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": specification,
        "decision": (
            {
                "analysis_id": decision_manifest.analysis_id,
                "path": str(decision_manifest.path),
                "content_hash": decision_manifest.content_hash,
            }
            if decision_manifest is not None
            else None
        ),
        "results": results,
        "report": str(report_path),
    }
    (analysis_root / "group" / "one_sample_group_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {"analysis": "one_sample", "report": str(report_path), "results": results}
