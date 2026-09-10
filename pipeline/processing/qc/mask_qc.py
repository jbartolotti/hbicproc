from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.plotting import plot_stat_map, view_img

from ...core.paths import write_json
from .base import QCReport, ReportResult
from .registry import register_report

logger = logging.getLogger(__name__)
_ENTITY_PATTERN = re.compile(r"(?:^|_)((?:sub|ses|task|run)-[^_]+)")
_MASK_PATTERN = re.compile(r"(?:^|_)(?P<subject>sub-[^_]+)(?:_(?P<session>ses-[^_]+))?_(?P<task>task-[^_]+)(?:_(?P<run>run-[^_]+))?_desc-mask$")
_DEFAULT_RARE_THRESHOLD_PCT = 10.0
_DEFAULT_MONTAGE_SLICES = 12


@dataclass(frozen=True)
class MaskRecord:
    path: Path
    subject: str
    session: str | None
    task: str
    run: str | None

    @property
    def session_task(self) -> tuple[str, str]:
        return (self.session or "n/a", self.task)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _entity_value(value: str | None, prefix: str) -> str | None:
    if value is None:
        return None
    return value[len(prefix) :] if value.startswith(prefix) else value


def _parse_mask(path: Path) -> MaskRecord | None:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    match = _MASK_PATTERN.match(name)
    if not match:
        return None
    return MaskRecord(
        path=path,
        subject=_entity_value(match.group("subject"), "sub-") or match.group("subject"),
        session=_entity_value(match.group("session"), "ses-"),
        task=_entity_value(match.group("task"), "task-") or match.group("task"),
        run=_entity_value(match.group("run"), "run-"),
    )


def _discover_masks(root: Path) -> tuple[list[MaskRecord], list[str]]:
    records: list[MaskRecord] = []
    warnings: list[str] = []
    candidates = sorted(root.rglob("*_desc-mask.nii.gz")) + sorted(root.rglob("*_desc-mask.nii"))
    for path in candidates:
        if "/models/" not in path.as_posix() and "\\models\\" not in path.as_posix():
            continue
        record = _parse_mask(path)
        if record is None:
            warnings.append(f"Could not parse BIDS entities from mask path: {path}")
            continue
        records.append(record)
    return records, warnings


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(config.get(key, default))
        if value <= 0 or value > 100:
            raise ValueError
        return value
    except (TypeError, ValueError):
        logger.warning("Invalid mask_qc %s=%r; using %.1f", key, config.get(key), default)
        return default


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key, default))
        if value < 3:
            raise ValueError
        return value
    except (TypeError, ValueError):
        logger.warning("Invalid mask_qc %s=%r; using %d", key, config.get(key), default)
        return default


def _load_masks(records: list[MaskRecord]) -> tuple[list[MaskRecord], list[np.ndarray], list[tuple[int, ...]], list[str]]:
    valid_records: list[MaskRecord] = []
    arrays: list[np.ndarray] = []
    shapes: list[tuple[int, ...]] = []
    warnings: list[str] = []
    for record in records:
        try:
            image = nib.load(str(record.path))
            data = np.asarray(image.get_fdata())
            if data.ndim != 3:
                warnings.append(f"Mask is not 3D and was skipped: {record.path}")
                continue
            arrays.append(np.isfinite(data) & (data > 0))
            shapes.append(data.shape)
            valid_records.append(record)
        except Exception as exc:
            warnings.append(f"Could not load mask {record.path}: {exc}")
    return valid_records, arrays, shapes, warnings


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value).strip())


def _label(session: str | None, task: str) -> str:
    return f"ses-{_safe_component(session or 'n-a')}_task-{_safe_component(task)}"


def _volume_rows(records: list[MaskRecord], arrays: list[np.ndarray], voxel_volume: float) -> pd.DataFrame:
    rows = []
    for record, mask in zip(records, arrays):
        voxels = int(mask.sum())
        rows.append(
            {
                "subject": record.subject,
                "session": record.session or "n/a",
                "task": record.task,
                "run": record.run or "n/a",
                "mask_path": str(record.path),
                "voxel_count": voxels,
                "mask_volume_mm3": float(voxels * voxel_volume),
                "mask_volume_ml": float(voxels * voxel_volume / 1000.0),
            }
        )
    return pd.DataFrame(rows)


def _coverage_map(arrays: list[np.ndarray]) -> np.ndarray:
    return np.sum(np.stack(arrays, axis=0), axis=0, dtype=np.int32)


def _rare_scores(arrays: list[np.ndarray], coverage: np.ndarray, threshold_pct: float) -> np.ndarray:
    threshold = len(arrays) * threshold_pct / 100.0
    rare = coverage > 0
    rare &= coverage < threshold
    return np.asarray([int(np.count_nonzero(mask & rare)) for mask in arrays], dtype=np.int64)


def _histogram(volume_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = volume_frame["mask_volume_ml"].astype(float)
    figure, axis = plt.subplots(figsize=(7, 4))
    bins = max(5, min(20, len(values)))
    axis.hist(values, bins=bins, edgecolor="black", color="#4c78a8")
    axis.axvline(values.mean(), color="#e45756", linestyle="--", label=f"Mean {values.mean():.2f} mL")
    axis.axvline(values.median(), color="#f2cf5b", linestyle="-.", label=f"Median {values.median():.2f} mL")
    axis.set_xlabel("Mask volume (mL)")
    axis.set_ylabel("Number of masks")
    axis.set_title("Mask volume distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _montage(record: MaskRecord, mask: np.ndarray, reference: nib.Nifti1Image, output_path: Path, slices: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = nib.Nifti1Image(mask.astype(np.float32), reference.affine, reference.header)
    coordinates = np.linspace(0.12, 0.88, slices)
    figure = plt.figure(figsize=(slices * 1.2, 1.7))
    display = plot_stat_map(image, bg_img="MNI152", display_mode="z", cut_coords=coordinates, threshold=0.5, colorbar=False, figure=figure)
    display.title(f"{record.subject} run-{record.run or 'n/a'}", size=8)
    figure.tight_layout(pad=0.2)
    figure.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No records available.</p>"
    return frame.to_html(index=False, border=0, classes="data-table", escape=True, float_format=lambda value: f"{value:.3f}")


def _write_combo_report(
    combo_dir: Path,
    session: str,
    task: str,
    records: list[MaskRecord],
    arrays: list[np.ndarray],
    threshold_pct: float,
    montage_slices: int,
    warnings: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    combo_dir.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(records[0].path))
    voxel_volume = abs(float(np.linalg.det(reference.affine[:3, :3])))
    logger.info("Calculating mask volumes: session=%s task=%s masks=%d", session, task, len(records))
    volumes = _volume_rows(records, arrays, voxel_volume)
    coverage = _coverage_map(arrays)
    logger.info("Generating mask coverage: session=%s task=%s shape=%s", session, task, coverage.shape)
    coverage_image = nib.Nifti1Image(coverage.astype(np.int16), reference.affine, reference.header)
    coverage_path = combo_dir / f"{_label(session, task)}_mask_coverage.nii.gz"
    coverage_image.to_filename(coverage_path)
    rare_scores = _rare_scores(arrays, coverage, threshold_pct)
    logger.info("Calculating rare voxel scores: session=%s task=%s threshold_pct=%.1f", session, task, threshold_pct)
    volumes["rare_voxel_score"] = rare_scores
    ranking = volumes.sort_values(["rare_voxel_score", "mask_volume_ml"], ascending=[False, False]).reset_index(drop=True)
    histogram_path = combo_dir / "mask_volume_histogram.png"
    _histogram(volumes, histogram_path)
    montage_dir = combo_dir / "montages"
    montage_dir.mkdir(exist_ok=True)
    montage_paths: dict[str, str] = {}
    logger.info("Generating %d subject mask montages: session=%s task=%s", len(records), session, task)
    for record, mask in zip(records, arrays):
        montage_path = montage_dir / f"{_safe_component(record.subject)}_run-{_safe_component(record.run or 'n-a')}_mask.png"
        try:
            _montage(record, mask, reference, montage_path, montage_slices)
            montage_paths[str(record.path)] = str(montage_path.relative_to(combo_dir))
        except Exception as exc:
            warnings.append(f"Could not generate montage for {record.path}: {exc}")
            logger.exception("Could not generate mask montage for %s", record.path)

    viewer_path = combo_dir / f"{_label(session, task)}_coverage_viewer.html"
    try:
        logger.info("Generating interactive coverage viewer: session=%s task=%s", session, task)
        view_img(str(coverage_path), title=f"Mask coverage: ses-{session} task-{task}", threshold=0, cmap="viridis", symmetric_cmap=False).save_as_html(str(viewer_path))
    except Exception as exc:
        warnings.append(f"Could not generate coverage viewer for {session}/{task}: {exc}")
        logger.exception("Could not generate coverage viewer for %s/%s", session, task)

    volumes_for_report = ranking.copy()
    volumes_for_report["mask_volume_ml"] = volumes_for_report["mask_volume_ml"].round(3)
    summary = {
        "number_of_masks": int(len(records)),
        "mean_mask_volume_ml": float(volumes["mask_volume_ml"].mean()),
        "median_mask_volume_ml": float(volumes["mask_volume_ml"].median()),
        "minimum_mask_volume_ml": float(volumes["mask_volume_ml"].min()),
        "maximum_mask_volume_ml": float(volumes["mask_volume_ml"].max()),
        "standard_deviation_mask_volume_ml": float(volumes["mask_volume_ml"].std(ddof=1)) if len(volumes) > 1 else 0.0,
        "voxel_volume_mm3": voxel_volume,
        "coverage_maximum": int(coverage.max()),
    }
    cards = []
    for _, row in ranking.iterrows():
        montage = montage_paths.get(str(row["mask_path"]))
        image = f"<img class='montage' src='{html.escape(montage)}' alt='Mask montage for {html.escape(str(row['subject']))}'>" if montage else "<p>Montage unavailable.</p>"
        cards.append(
            "<article class='subject-card'>"
            f"<h3>{html.escape(str(row['subject']))} <small>run-{html.escape(str(row['run']))}</small></h3>"
            f"<p>Mask volume: {row['mask_volume_ml']:.3f} mL | Rare voxel score: {int(row['rare_voxel_score'])}</p>{image}</article>"
        )
    summary_items = "".join(f"<li><strong>{html.escape(key)}:</strong> {value}</li>" for key, value in summary.items())
    viewer = f"<iframe class='viewer' src='{html.escape(viewer_path.name)}' title='Interactive mask coverage viewer'></iframe>" if viewer_path.exists() else "<p>Interactive viewer unavailable.</p>"
    report_html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Mask QC: ses-{html.escape(session)} task-{html.escape(task)}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;color:#222}.data-table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.85rem}.data-table th,.data-table td{border:1px solid #ccc;padding:.35rem;text-align:left}.data-table th{background:#eee}.viewer{width:100%;height:650px;border:1px solid #bbb}.histogram{max-width:800px;width:100%}.subject-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}.subject-card{border:1px solid #ccc;padding:.75rem}.montage{max-width:100%;height:auto}small{font-weight:normal;color:#666}</style></head><body>"
        f"<h1>Mask QC</h1><h2>Session: {html.escape(session)} | Task: {html.escape(task)}</h2>"
        "<h2>Overview</h2><ul>" + summary_items + "</ul>"
        + "<h2>Coverage Map Viewer</h2>" + viewer
        + "<p>Coverage intensity is the number of analyzed masks containing each voxel.</p>"
        + "<h2>Coverage Summary Statistics</h2>" + _html_table(pd.DataFrame([summary]))
        + "<h2>Mask Volume Histogram</h2><img class='histogram' src='mask_volume_histogram.png' alt='Mask volume histogram'>"
        + "<h2>Rare Voxel Ranking Table</h2>" + _html_table(ranking[["subject", "session", "task", "run", "mask_volume_ml", "rare_voxel_score"]])
        + "<h2>Subject Review Cards</h2><div class='subject-grid'>" + "".join(cards) + "</div>"
        + "</body></html>\n"
    )
    report_path = combo_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    metadata = {
        "report_type": "mask_qc",
        "report_version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "task": task,
        "number_of_masks_analyzed": len(records),
        "rare_voxel_threshold_pct": threshold_pct,
        "montage_slices": montage_slices,
        "summary": summary,
        "source_files": [str(record.path) for record in records],
        "warnings": warnings,
        "software": {"hbicproc": _package_version("hbicproc-pipeline"), "nilearn": _package_version("nilearn"), "nibabel": _package_version("nibabel")},
        "outputs": [str(report_path), str(coverage_path), str(histogram_path), str(viewer_path), *[str(combo_dir / path) for path in montage_paths.values()]],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(combo_dir / "metadata.json", metadata)
    return {"session": session, "task": task, "report": str(report_path), "metadata": metadata}


@register_report
class MaskQCReport(QCReport):
    name = "mask_qc"
    scope = "dataset"

    def generate(self, context) -> ReportResult:
        started = time.perf_counter()
        output_dir = context.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        report_config = context.report_config
        threshold_pct = _config_float(report_config, "rare_voxel_threshold_pct", _DEFAULT_RARE_THRESHOLD_PCT)
        montage_slices = _config_int(report_config, "montage_slices", _DEFAULT_MONTAGE_SLICES)
        analysis_root = Path(context.config["analysis"]["output_dir"])
        logger.info("Initializing mask_qc: output=%s analysis_root=%s", output_dir, analysis_root)
        records, warnings = _discover_masks(analysis_root)
        logger.info("Discovered %d first-level masks", len(records))
        if not records:
            return ReportResult(self.name, "skipped", "mask_qc skipped: no first-level desc-mask derivatives were found.", output_dir, {"warnings": warnings})

        groups: dict[tuple[str, str], list[MaskRecord]] = {}
        for record in records:
            groups.setdefault(record.session_task, []).append(record)
        combo_results = []
        for (session, task), combo_records in sorted(groups.items()):
            logger.info("Processing mask group: session=%s task=%s masks=%d", session, task, len(combo_records))
            valid_records, arrays, shapes, load_warnings = _load_masks(combo_records)
            warnings.extend(load_warnings)
            if not arrays:
                warnings.append(f"No readable masks for session={session}, task={task}")
                continue
            if len(set(shapes)) != 1:
                warnings.append(f"Masks with inconsistent shapes skipped for session={session}, task={task}")
                shape = max(set(shapes), key=shapes.count)
                keep = [index for index, current_shape in enumerate(shapes) if current_shape == shape]
                valid_records = [valid_records[index] for index in keep]
                arrays = [arrays[index] for index in keep]
            combo_dir = output_dir / f"ses-{_safe_component(session)}" / f"task-{_safe_component(task)}"
            combo_results.append(_write_combo_report(combo_dir, session, task, valid_records, arrays, threshold_pct, montage_slices, warnings))

        links = "".join(
            f"<tr><td>{html.escape(result['session'])}</td><td>{html.escape(result['task'])}</td><td>{result['metadata']['number_of_masks_analyzed']}</td><td><a href='{html.escape(Path(result['report']).relative_to(output_dir).as_posix())}'>open report</a></td></tr>"
            for result in combo_results
        )
        report_path = output_dir / "report.html"
        report_path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>Mask QC</title></head><body>"
            "<h1>Mask QC</h1><p>Population-level first-level mask consistency reports.</p>"
            "<table border='1'><tr><th>Session</th><th>Task</th><th>Masks</th><th>Report</th></tr>" + links + "</table></body></html>\n",
            encoding="utf-8",
        )
        metadata = {
            "report_type": self.name,
            "report_version": "1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "number_of_masks_analyzed": sum(result["metadata"]["number_of_masks_analyzed"] for result in combo_results),
            "session_task_combinations": [{"session": result["session"], "task": result["task"], "number_of_masks": result["metadata"]["number_of_masks_analyzed"]} for result in combo_results],
            "rare_voxel_threshold_pct": threshold_pct,
            "montage_slices": montage_slices,
            "warnings": warnings,
            "software": {"hbicproc": _package_version("hbicproc-pipeline")},
            "outputs": [str(report_path), *[result["report"] for result in combo_results]],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(output_dir / "metadata.json", metadata)
        logger.info("Generated mask_qc reports for %d session/task combinations in %.2f seconds", len(combo_results), metadata["elapsed_seconds"])
        return ReportResult(self.name, "completed", f"mask_qc generated {len(combo_results)} session/task reports: {report_path}", output_dir, {"warnings": warnings, "combinations": metadata["session_task_combinations"]})
