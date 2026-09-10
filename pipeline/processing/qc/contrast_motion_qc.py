from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import new_img_like
from nilearn.plotting import html_stat_map

from ...core.paths import write_json
from .base import QCReport, ReportResult
from .motion_qc import load_motion_summary
from .registry import register_report

logger = logging.getLogger(__name__)
_CONTRAST_FILE_PATTERN = re.compile(r"desc-atlas-(?P<atlas>.+)-contrasts\.tsv$")
_ENTITY_PATTERN = re.compile(r"(?:^|_)((?:sub|ses|task|run)-[^_]+)")
_DEFAULT_METRICS = ("mean_fd", "median_fd", "pct_fd_02", "pct_fd_05")
_DEFAULT_THRESHOLDS = {"warning": 0.20, "severe": 0.30}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _entities(path: Path) -> dict[str, str | None]:
    name = path.name
    for suffix in (".tsv", ".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    values = {"subject": None, "session": None, "task": None, "run": None}
    for match in _ENTITY_PATTERN.finditer(name):
        key, value = match.group(1).split("-", 1)
        values[{"sub": "subject", "ses": "session"}.get(key, key)] = value
    return values


def _configured_list(value: Any, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else list(default)
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    return list(default)


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("parcel_thresholds", {})
    raw = raw if isinstance(raw, dict) else {}
    values = dict(_DEFAULT_THRESHOLDS)
    for key in values:
        try:
            candidate = float(raw.get(key, values[key]))
            if candidate < 0:
                raise ValueError
            values[key] = candidate
        except (TypeError, ValueError):
            logger.warning("Invalid contrast_motion_qc parcel_thresholds.%s; using %.2f", key, values[key])
    if values["warning"] > values["severe"]:
        logger.warning("contrast_motion_qc warning threshold exceeds severe threshold; using defaults")
        return dict(_DEFAULT_THRESHOLDS)
    return values


def _atlas_files(output_root: Path, requested: list[str]) -> list[tuple[str, Path]]:
    discovered = []
    for path in output_root.rglob("*_desc-atlas-*-contrasts.tsv"):
        match = _CONTRAST_FILE_PATTERN.search(path.name)
        if not match:
            continue
        atlas = match.group("atlas")
        if requested and atlas not in requested:
            continue
        discovered.append((atlas, path))
    return sorted(discovered, key=lambda item: (item[0], str(item[1])))


def _contrast_rows(files: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows = []
    source_files = []
    warnings = []
    for atlas, path in files:
        try:
            table = pd.read_csv(path, sep="\t")
        except Exception as exc:
            warnings.append(f"Could not read atlas contrast summary {path}: {exc}")
            continue
        required = {"parcel_id", "contrast", "effect"}
        missing = required - set(table.columns)
        if missing:
            warnings.append(f"Atlas contrast summary {path} is missing columns: {sorted(missing)}")
            continue
        entities = _entities(path)
        table = table.copy()
        table["atlas"] = atlas
        table["source_path"] = str(path)
        for key, value in entities.items():
            table[key] = value
        table["parcel_id"] = pd.to_numeric(table["parcel_id"], errors="coerce")
        table["effect"] = pd.to_numeric(table["effect"], errors="coerce")
        table = table.dropna(subset=["parcel_id", "effect", "contrast", "subject"])
        if not table.empty:
            rows.append(table)
            source_files.append(str(path))
    if not rows:
        return pd.DataFrame(), source_files, warnings
    return pd.concat(rows, ignore_index=True), source_files, warnings


def _correlation(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    values = pd.concat([x, y], axis=1).dropna()
    n = len(values)
    if n < 3 or values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return float("nan"), n
    return float(values.iloc[:, 0].corr(values.iloc[:, 1], method="pearson")), n


def _summary(values: pd.Series, thresholds: dict[str, float]) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    absolute = values.abs()
    if values.empty:
        return {
            "mean_correlation": None,
            "median_correlation": None,
            "mean_absolute_correlation": None,
            "median_absolute_correlation": None,
            "standard_deviation": None,
            "95th_percentile_absolute_correlation": None,
            "maximum_absolute_correlation": None,
            "n_parcels": 0,
            "n_parcels_abs_r_gt_warning": 0,
            "n_parcels_abs_r_gt_severe": 0,
        }
    return {
        "mean_correlation": float(values.mean()),
        "median_correlation": float(values.median()),
        "mean_absolute_correlation": float(absolute.mean()),
        "median_absolute_correlation": float(absolute.median()),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "95th_percentile_absolute_correlation": float(absolute.quantile(0.95)),
        "maximum_absolute_correlation": float(absolute.max()),
        "n_parcels": int(len(values)),
        "n_parcels_abs_r_gt_warning": int((absolute > thresholds["warning"]).sum()),
        "n_parcels_abs_r_gt_severe": int((absolute > thresholds["severe"]).sum()),
    }


def _interpret(summary: dict[str, Any]) -> str:
    if not summary["n_parcels"]:
        return "No parcel correlations could be computed."
    return (
        f"Median absolute parcel correlation was {summary['median_absolute_correlation']:.2f}. "
        f"{summary['n_parcels_abs_r_gt_severe']} of {summary['n_parcels']} parcels exceeded "
        f"|r| > {summary.get('severe_threshold', _DEFAULT_THRESHOLDS['severe']):.2f}."
    )


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No results available.</p>"
    return frame.to_html(index=False, border=0, classes="data-table", escape=True, float_format=lambda value: f"{value:.4f}")


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value).strip())


def _map_and_viewer(
    correlations: pd.DataFrame,
    atlas_path: Path,
    output_dir: Path,
    stem: str,
) -> tuple[str | None, str | None]:
    if not atlas_path.exists():
        return None, None
    try:
        atlas_image = nib.load(str(atlas_path))
        data = np.asarray(atlas_image.get_fdata())
        correlation_by_parcel = correlations.set_index("parcel_id")["correlation"].to_dict()
        map_data = np.zeros(data.shape, dtype=float)
        for parcel_id, value in correlation_by_parcel.items():
            if pd.notna(value):
                map_data[np.rint(data).astype(np.int64) == int(parcel_id)] = float(value)
        map_path = output_dir / "maps" / f"{stem}_correlation.nii.gz"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        new_img_like(atlas_image, map_data).to_filename(map_path)
        viewer_path = output_dir / "viewers" / f"{stem}_correlation.html"
        viewer_path.parent.mkdir(parents=True, exist_ok=True)
        viewer = html_stat_map(str(map_path), title=stem, threshold=0, cmap="RdBu_r", symmetric_cmap=True)
        viewer.save_as_html(str(viewer_path))
        return str(map_path), str(viewer_path)
    except Exception:
        logger.exception("Could not create correlation map/viewer for %s", stem)
        return None, None


@register_report
class ContrastMotionQCReport(QCReport):
    name = "contrast_motion_qc"
    scope = "dataset"

    def generate(self, context) -> ReportResult:
        started = time.perf_counter()
        output_dir = context.output_dir / self.name
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        report_config = context.report_config
        metrics = _configured_list(report_config.get("motion_metrics"), _DEFAULT_METRICS)
        thresholds = _thresholds(report_config)
        requested_atlases = _configured_list(report_config.get("atlases"), ())
        analysis_root = Path(context.config["analysis"]["output_dir"])
        logger.info("Initializing contrast_motion_qc: output=%s metrics=%s", output_dir, metrics)
        logger.info("Discovering atlas contrast summaries under analysis output root %s", analysis_root)
        atlas_files = _atlas_files(analysis_root, requested_atlases)
        logger.info("Discovered %d atlas contrast summary files across atlases=%s", len(atlas_files), sorted({item[0] for item in atlas_files}))
        contrast_data, contrast_sources, warnings = _contrast_rows(atlas_files)
        motion_data, motion_sources, motion_warnings = load_motion_summary(context, context.report_config)
        warnings.extend(motion_warnings)
        if contrast_data.empty or motion_data.empty:
            return ReportResult(self.name, "skipped", "contrast_motion_qc skipped: atlas contrast summaries or motion metrics were unavailable.", output_dir, {"warnings": warnings})

        motion_by_subject = motion_data.groupby("subject", dropna=False)[metrics].mean().reset_index()
        summary_rows = []
        network_rows = []
        histogram_files = []
        viewer_rows = []
        detail_sections = []
        discovered_contrasts = sorted(contrast_data["contrast"].astype(str).unique())
        logger.info("Discovered contrasts: %s", discovered_contrasts)
        for atlas in sorted(contrast_data["atlas"].unique()):
            atlas_data = contrast_data[contrast_data["atlas"] == atlas]
            atlas_summary_sections = []
            atlas_path_by_file = {}
            for source_path in atlas_data["source_path"].unique():
                path = Path(source_path)
                atlas_path_by_file[source_path] = path.with_name(path.name.replace("-contrasts.tsv", "-resampled.nii.gz"))
            for contrast in sorted(atlas_data["contrast"].astype(str).unique()):
                contrast_data_one = atlas_data[atlas_data["contrast"] == contrast]
                metric_sections = []
                for metric in metrics:
                    parcel_rows = []
                    for parcel_id, parcel_data in contrast_data_one.groupby("parcel_id"):
                        subject_effect = parcel_data.groupby("subject", as_index=False)["effect"].mean()
                        merged = subject_effect.merge(motion_by_subject[["subject", metric]], on="subject", how="inner")
                        correlation, n = _correlation(merged[metric], merged["effect"])
                        label = str(parcel_data["parcel_label"].dropna().iloc[0]) if "parcel_label" in parcel_data and not parcel_data["parcel_label"].dropna().empty else str(parcel_id)
                        network = str(parcel_data["network"].dropna().iloc[0]) if "network" in parcel_data and not parcel_data["network"].dropna().empty else ""
                        parcel_rows.append({"parcel_id": int(parcel_id), "parcel_label": label, "network": network, "correlation": correlation, "n_subjects": n})
                    parcel_frame = pd.DataFrame(parcel_rows).dropna(subset=["correlation"])
                    stats = _summary(parcel_frame["correlation"], thresholds)
                    stats.update({"atlas": atlas, "contrast": contrast, "motion_metric": metric, "warning_threshold": thresholds["warning"], "severe_threshold": thresholds["severe"]})
                    summary_rows.append(stats)
                    flag = stats["95th_percentile_absolute_correlation"] is not None and (
                        stats["95th_percentile_absolute_correlation"] > thresholds["severe"]
                        or stats["n_parcels_abs_r_gt_severe"] > 0.10 * stats["n_parcels"]
                    )
                    stats["flagged"] = bool(flag)
                    logger.info("Computed parcel correlations: atlas=%s contrast=%s metric=%s parcels=%d", atlas, contrast, metric, len(parcel_frame))
                    figure = _histogram(parcel_frame["correlation"], atlas, contrast, metric, figure_dir, thresholds)
                    histogram_files.append(figure)
                    source_path = str(contrast_data_one["source_path"].iloc[0])
                    map_stem = f"{_safe_component(atlas)}_{_safe_component(contrast)}_{_safe_component(metric)}"
                    map_path, viewer_path = _map_and_viewer(parcel_frame, atlas_path_by_file.get(source_path, Path()), output_dir, map_stem)
                    viewer_rows.append({"atlas": atlas, "contrast": contrast, "motion_metric": metric, "viewer": viewer_path, "map": map_path})
                    if network_rows is not None and "network" in parcel_frame and parcel_frame["network"].astype(str).str.strip().any():
                        network_effect = contrast_data_one[contrast_data_one["network"].astype(str).str.strip() != ""].groupby(["subject", "network"], as_index=False)["effect"].mean()
                        for network, network_data in network_effect.groupby("network"):
                            merged = network_data.merge(motion_by_subject[["subject", metric]], on="subject", how="inner")
                            correlation, n = _correlation(merged[metric], merged["effect"])
                            network_rows.append({"network": network, "correlation": correlation, "sample_size": n, "motion_metric": metric, "atlas": atlas, "contrast": contrast})
                    metric_sections.append((metric, stats, parcel_frame))
                atlas_summary_sections.append((contrast, metric_sections))
            detail_sections.append((atlas, atlas_summary_sections))

        summary_frame = pd.DataFrame(summary_rows)
        network_frame = pd.DataFrame(network_rows)
        overview_lines = []
        for row in summary_rows:
            overview_lines.append(f"<h4>{html.escape(row['atlas'])} / {html.escape(row['contrast'])} / {html.escape(row['motion_metric'])}</h4><p>{html.escape(_interpret(row))}</p>")
        viewer_html = "".join(
            f"<h4>{html.escape(row['atlas'])} / {html.escape(row['contrast'])} / {html.escape(row['motion_metric'])}</h4>"
            + (f"<iframe class='viewer' src='../{html.escape(Path(row['viewer']).relative_to(output_dir).as_posix())}'></iframe>" if row["viewer"] else "<p>Viewer unavailable.</p>")
            for row in viewer_rows
        )
        detail_html = "".join(
            f"<details><summary>{html.escape(atlas)} / {html.escape(contrast)}</summary>" + "".join(
                f"<h4>{html.escape(metric)}</h4>{_html_table(parcel_frame)}" for metric, _, parcel_frame in metric_sections
            ) + "</details>"
            for atlas, contrast_sections in detail_sections for contrast, metric_sections in contrast_sections
        )
        report_html = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Contrast Motion QC</title>"
            "<style>body{font-family:Arial,sans-serif;margin:2rem;color:#222}.data-table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.85rem}.data-table th,.data-table td{border:1px solid #ccc;padding:.3rem;text-align:left}.data-table th{background:#eee}.histogram{max-width:800px;width:100%}.viewer{width:100%;height:620px;border:1px solid #bbb}details{margin:1rem 0}</style></head><body>"
            "<h1>Contrast Motion QC</h1><h2>Report Overview</h2>"
            f"<p>Evaluated {len(summary_frame)} atlas/contrast/motion combinations using {motion_data['subject'].nunique()} subjects and {len(motion_data)} motion runs. Flags are review recommendations only.</p>"
            + "".join(overview_lines)
            + "<h2>Atlas / Contrast / Metric Summary</h2>" + _html_table(summary_frame)
            + "<h2>Network-Level Results</h2>" + _html_table(network_frame)
            + "<h2>Parcel Correlation Histograms</h2>" + "".join(f"<h4>{html.escape(Path(path).stem)}</h4><img class='histogram' src='figures/{html.escape(Path(path).name)}'>" for path in histogram_files)
            + "<h2>Interactive Brain Correlation Maps</h2>" + viewer_html
            + "<h2>Detailed Parcel Results</h2>" + detail_html
            + "</body></html>\n"
        )
        report_path = output_dir / "report.html"
        report_path.write_text(report_html, encoding="utf-8")
        elapsed = round(time.perf_counter() - started, 3)
        metadata = {
            "report_type": self.name,
            "report_version": "1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "configuration": report_config,
            "atlases": sorted(contrast_data["atlas"].unique()),
            "contrasts": discovered_contrasts,
            "motion_metrics": metrics,
            "subject_count": int(motion_data["subject"].nunique()),
            "run_count": int(len(motion_data)),
            "source_files": sorted(set(contrast_sources + motion_sources)),
            "warnings": warnings,
            "software": {"hbicproc": _package_version("hbicproc-pipeline"), "pandas": _package_version("pandas"), "nilearn": _package_version("nilearn")},
            "outputs": [str(report_path), *histogram_files, *[row["map"] for row in viewer_rows if row["map"]], *[row["viewer"] for row in viewer_rows if row["viewer"]]],
            "elapsed_seconds": elapsed,
        }
        write_json(output_dir / "metadata.json", metadata)
        logger.info("Wrote contrast_motion_qc report in %.2f seconds", elapsed)
        return ReportResult(self.name, "completed", f"contrast_motion_qc report generated: {report_path}", output_dir, {"summary": metadata["atlases"], "warnings": warnings})


def _histogram(values: pd.Series, atlas: str, contrast: str, metric: str, figure_dir: Path, thresholds: dict[str, float]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = figure_dir / f"{_safe_component(atlas)}_{_safe_component(contrast)}_{_safe_component(metric)}_correlations.png"
    figure, axis = plt.subplots(figsize=(7, 4))
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        values = pd.Series([0.0])
    minimum, maximum = float(values.min()), float(values.max())
    span = max(maximum - minimum, 0.1)
    axis.hist(values, bins=np.linspace(minimum - span * 0.05, maximum + span * 0.05, 11), edgecolor="black")
    axis.axvline(0, color="black")
    axis.axvline(thresholds["warning"], color="darkorange", linestyle="--")
    axis.axvline(-thresholds["warning"], color="darkorange", linestyle="--")
    axis.axvline(thresholds["severe"], color="red", linestyle="--")
    axis.axvline(-thresholds["severe"], color="red", linestyle="--")
    axis.set_xlabel("Parcel-wise Pearson r")
    axis.set_ylabel("Number of parcels")
    axis.set_title(f"{atlas} / {contrast} / {metric}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)
