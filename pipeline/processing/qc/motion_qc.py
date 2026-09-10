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

import pandas as pd

from ...core.paths import write_json
from ..analysis.context import InputDataset
from .base import QCReport, ReportResult
from .registry import register_report

logger = logging.getLogger(__name__)
_ENTITY_PATTERN = re.compile(r"(?:^|_)((?:sub|ses|task|run)-[^_]+)")
_DEFAULT_THRESHOLDS = {"warning": 0.2, "severe": 0.5}


@dataclass
class RunMotion:
    subject: str
    session: str | None
    task: str | None
    run: str | None
    source_path: Path
    fd: list[float]

    @property
    def mean_fd(self) -> float:
        return float(pd.Series(self.fd).mean())

    @property
    def median_fd(self) -> float:
        return float(pd.Series(self.fd).median())

    @property
    def max_fd(self) -> float:
        return float(max(self.fd)) if self.fd else 0.0

    @property
    def n_volumes(self) -> int:
        return len(self.fd)

    def count_above(self, threshold: float) -> int:
        return sum(value > threshold for value in self.fd)

    def pct_above(self, threshold: float) -> float:
        return 100.0 * self.count_above(threshold) / self.n_volumes if self.n_volumes else 0.0

    def as_dict(self, thresholds: dict[str, float]) -> dict[str, Any]:
        warning = thresholds["warning"]
        severe = thresholds["severe"]
        return {
            "subject": self.subject,
            "session": self.session,
            "task": self.task,
            "run": self.run,
            "source_path": str(self.source_path),
            "mean_fd": self.mean_fd,
            "median_fd": self.median_fd,
            "max_fd": self.max_fd,
            "n_volumes": self.n_volumes,
            "n_fd_02": self.count_above(warning),
            "pct_fd_02": self.pct_above(warning),
            "n_fd_05": self.count_above(severe),
            "pct_fd_05": self.pct_above(severe),
        }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _entities_from_path(path: Path) -> dict[str, str | None]:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".tsv", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    entities: dict[str, str | None] = {"subject": None, "session": None, "task": None, "run": None}
    for match in _ENTITY_PATTERN.finditer(name):
        key, value = match.group(1).split("-", 1)
        entities[{"sub": "subject", "ses": "session"}.get(key, key)] = value
    return entities


def _label(value: str | None) -> str:
    return value if value not in (None, "") else "n/a"


def _bids_label(value: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", _label(value))


def _run_filename(run: RunMotion, suffix: str) -> str:
    return (
        f"sub-{_bids_label(run.subject)}_ses-{_bids_label(run.session)}_"
        f"task-{_bids_label(run.task)}_run-{_bids_label(run.run)}_{suffix}.png"
    )


def _thresholds(report_config: dict[str, Any]) -> dict[str, float]:
    configured = report_config.get("fd_thresholds", {})
    if not isinstance(configured, dict):
        configured = {}
    values = dict(_DEFAULT_THRESHOLDS)
    for name in values:
        try:
            candidate = float(configured.get(name, values[name]))
            if candidate >= 0:
                values[name] = candidate
        except (TypeError, ValueError):
            logger.warning("Invalid motion_qc fd_thresholds.%s; using %.3f", name, values[name])
    if values["warning"] > values["severe"]:
        logger.warning("motion_qc warning threshold exceeds severe threshold; using defaults")
        return dict(_DEFAULT_THRESHOLDS)
    return values


def _html_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "<p>No records available.</p>"
    selected = frame[columns] if columns else frame
    return selected.to_html(index=False, border=0, classes="data-table", escape=True, float_format=lambda value: f"{value:.4f}")


def _run_table(runs: list[RunMotion], thresholds: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([run.as_dict(thresholds) for run in runs])


def _flagged_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    warning = (frame["mean_fd"] >= 0.15) | (frame["pct_fd_02"] >= 10)
    severe = (frame["mean_fd"] >= 0.30) | (frame["pct_fd_05"] >= 10)
    flagged = frame[warning].copy()
    flagged["flag_level"] = "Yellow"
    flagged.loc[severe.loc[flagged.index], "flag_level"] = "Red"
    flagged["_flag_order"] = flagged["flag_level"].map({"Red": 0, "Yellow": 1})
    return flagged.sort_values(["_flag_order", "mean_fd"], ascending=[True, False]).drop(columns=["_flag_order"])


def _save_histograms(frame: pd.DataFrame, figure_dir: Path, thresholds: dict[str, float]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = [
        ("mean_fd", "Mean FD (mm)", "Mean FD histogram", "mean_fd_histogram.png"),
        ("pct_fd_02", f"FD > {thresholds['warning']:.2f} (%)", "Percent FD above warning threshold", "pct_fd_02_histogram.png"),
        ("pct_fd_05", f"FD > {thresholds['severe']:.2f} (%)", "Percent FD above severe threshold", "pct_fd_05_histogram.png"),
    ]
    outputs = []
    for column, xlabel, title, filename in plots:
        figure, axis = plt.subplots(figsize=(7, 4))
        values = frame[column].astype(float)
        minimum, maximum = float(values.min()), float(values.max())
        span = max(maximum - minimum, 0.1)
        lower = minimum - span * 0.05
        upper = maximum + span * 0.05
        bins = [lower + (upper - lower) * index / 10 for index in range(11)]
        axis.hist(values, bins=bins, edgecolor="black")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of runs")
        axis.set_title(title)
        figure.tight_layout()
        path = figure_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        outputs.append(path.name)
    return outputs


def _save_run_figures(runs: list[RunMotion], figure_dir: Path, thresholds: dict[str, float]) -> list[tuple[RunMotion, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace_dir = figure_dir / "fd_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for run in runs:
        figure, axis = plt.subplots(figsize=(8, 3.5))
        axis.plot(range(len(run.fd)), run.fd, linewidth=0.8)
        axis.axhline(thresholds["warning"], color="darkorange", linestyle="--", label=f"> {thresholds['warning']:.2f} mm")
        axis.axhline(thresholds["severe"], color="red", linestyle="--", label=f"> {thresholds['severe']:.2f} mm")
        axis.set_xlabel("Volume index")
        axis.set_ylabel("Framewise displacement (mm)")
        axis.set_title(f"{run.subject} | session={_label(run.session)} | task={_label(run.task)} | run={_label(run.run)}")
        axis.legend(loc="upper right")
        figure.tight_layout()
        filename = _run_filename(run, "fdtrace")
        figure.savefig(trace_dir / filename, dpi=120)
        plt.close(figure)
        outputs.append((run, f"figures/fd_traces/{filename}"))
    return outputs


def _save_sorted_cohort_figures(runs: list[RunMotion], figure_dir: Path, thresholds: dict[str, float]) -> dict[tuple[str | None, str | None], str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str | None, str | None], list[RunMotion]] = {}
    for run in runs:
        grouped.setdefault((run.session, run.task), []).append(run)
    outputs = {}
    cohort_dir = figure_dir / "sorted_fd"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    for (session, task), cohort in sorted(grouped.items(), key=lambda item: (_label(item[0][0]), _label(item[0][1]))):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for run in cohort:
            sorted_fd = sorted(run.fd)
            percentiles = [100 * index / max(len(sorted_fd) - 1, 1) for index in range(len(sorted_fd))]
            label = f"{run.subject} run-{_label(run.run)}"
            axis.plot(percentiles, sorted_fd, alpha=0.45, linewidth=0.9, label=label)
        axis.axhline(thresholds["warning"], color="darkorange", linestyle="--")
        axis.axhline(thresholds["severe"], color="red", linestyle="--")
        axis.set_xticks(range(0, 101, 10))
        axis.set_xlabel("Percentile of run volumes")
        axis.set_ylabel("Framewise displacement (mm)")
        axis.set_title(f"Sorted FD: session={_label(session)}, task={_label(task)}")
        axis.grid(axis="x", alpha=0.3)
        if len(cohort) <= 20:
            axis.legend(fontsize="small", ncol=2)
        figure.tight_layout()
        filename = f"ses-{_bids_label(session)}_task-{_bids_label(task)}_sortedfd.png"
        figure.savefig(cohort_dir / filename, dpi=150)
        plt.close(figure)
        outputs[(session, task)] = f"figures/sorted_fd/{filename}"
    return outputs


def _summary_html(summary: dict[str, Any]) -> str:
    items = "".join(
        f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"
        for key, value in summary.items()
    )
    return f"<ul class='summary'>{items}</ul>"


@register_report
class MotionQCReport(QCReport):
    name = "motion_qc"
    scope = "dataset"

    def generate(self, context) -> ReportResult:
        started = time.perf_counter()
        output_dir = context.output_dir / self.name
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        report_config = context.report_config
        thresholds = _thresholds(report_config)
        logger.info("Initializing motion_qc report (thresholds=%s output=%s)", thresholds, output_dir)

        input_dataset_config = report_config.get("input_dataset") or context.config["analysis"]["input_dataset"]
        dataset = InputDataset(str(input_dataset_config["name"]), Path(input_dataset_config["path"]))
        logger.info("Discovering fMRIPrep confounds through the cached PyBIDS dataset index")
        confounds_files = context.dataset_index.get_confounds_files(input_dataset=dataset)
        logger.info("Discovered %d confounds files", len(confounds_files))
        runs: list[RunMotion] = []
        source_files: list[str] = []
        warnings: list[str] = []

        for path in confounds_files:
            entities = _entities_from_path(path)
            subject = entities["subject"]
            if subject is None:
                warnings.append(f"Could not determine subject from {path}")
                continue
            try:
                confounds = pd.read_csv(path, sep="\t")
            except Exception as exc:
                warnings.append(f"Could not read {path}: {exc}")
                continue
            if "framewise_displacement" not in confounds.columns:
                warnings.append(f"Missing framewise_displacement column in {path}")
                continue
            displacement = pd.to_numeric(confounds["framewise_displacement"], errors="coerce").fillna(0.0)
            values = displacement.astype(float).tolist()
            if not values:
                warnings.append(f"No framewise displacement volumes in {path}")
                continue
            runs.append(RunMotion(subject, entities["session"], entities["task"], entities["run"], path, values))
            source_files.append(str(path))
        logger.info("Extracted motion metrics for %d runs across %d subjects", len(runs), len({run.subject for run in runs}))

        if not runs:
            return ReportResult(
                self.name,
                "skipped",
                "motion_qc skipped: no usable fMRIPrep confounds files were found.",
                output_dir,
                {"warnings": warnings},
            )

        frame = _run_table(runs, thresholds).sort_values("mean_fd", ascending=False)
        all_fd = [value for run in runs for value in run.fd]
        summary = {
            "total subjects": frame["subject"].nunique(),
            "total runs": len(frame),
            "overall mean FD": round(float(pd.Series(all_fd).mean()), 4),
            "overall median FD": round(float(pd.Series(all_fd).median()), 4),
            "95th percentile mean FD": round(float(frame["mean_fd"].quantile(0.95)), 4),
            f"runs with >10% FD > {thresholds['warning']:.2f}": int((frame["pct_fd_02"] > 10).sum()),
            f"runs with >10% FD > {thresholds['severe']:.2f}": int((frame["pct_fd_05"] > 10).sum()),
        }

        logger.info("Generating motion histograms")
        histogram_files = _save_histograms(frame, figure_dir, thresholds)
        logger.info("Generating %d individual FD trace figures", len(runs))
        trace_files = _save_run_figures(runs, figure_dir, thresholds)
        logger.info("Generating session/task sorted FD cohort figures")
        cohort_files = _save_sorted_cohort_figures(runs, figure_dir, thresholds)

        grouped = frame.groupby(["session", "task"], dropna=False)
        session_task = grouped.agg(
            n_runs=("mean_fd", "size"),
            mean_mean_fd=("mean_fd", "mean"),
            median_mean_fd=("mean_fd", "median"),
            mean_pct_fd_02=("pct_fd_02", "mean"),
            mean_pct_fd_05=("pct_fd_05", "mean"),
        ).reset_index()
        session_task["session"] = session_task["session"].fillna("n/a")
        session_task["task"] = session_task["task"].fillna("n/a")
        flagged = _flagged_table(frame)

        trace_by_group: dict[tuple[str | None, str | None], list[tuple[RunMotion, str]]] = {}
        for run, relative_path in trace_files:
            trace_by_group.setdefault((run.session, run.task), []).append((run, relative_path))
        cohort_sections = "".join(
            f"<h3>Session: {html.escape(_label(session))} | Task: {html.escape(_label(task))}</h3>"
            f"<img class='cohort-figure' src='{html.escape(relative_path)}' alt='Sorted FD cohort'>"
            for (session, task), relative_path in sorted(cohort_files.items(), key=lambda item: (_label(item[0][0]), _label(item[0][1])))
        )
        trace_sections = []
        for (session, task), group_runs in sorted(trace_by_group.items(), key=lambda item: (_label(item[0][0]), _label(item[0][1]))):
            images = "".join(
                f"<figure><img class='trace-figure' src='{html.escape(relative_path)}' alt='FD trace for {html.escape(run.subject)}'>"
                f"<figcaption>{html.escape(run.subject)} | run={html.escape(_label(run.run))}</figcaption></figure>"
                for run, relative_path in group_runs
            )
            trace_sections.append(
                f"<h3>Session: {html.escape(_label(session))} | Task: {html.escape(_label(task))}</h3><div class='trace-grid'>{images}</div>"
            )

        report_path = output_dir / "report.html"
        report_html = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Motion QC</title>"
            "<style>body{font-family:Arial,sans-serif;margin:2rem;color:#222}.summary{line-height:1.8}"
            ".data-table{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.9rem}"
            ".data-table th,.data-table td{border:1px solid #ccc;padding:.35rem;text-align:left}"
            ".data-table th{background:#eee}.histogram,.cohort-figure{max-width:900px;width:100%;margin:1rem 0}"
            ".trace-grid{display:flex;flex-wrap:wrap;gap:1rem}.trace-grid figure{margin:0;width:31%;min-width:280px}"
            ".trace-figure{width:100%}figcaption{font-size:.85rem}</style></head><body>"
            "<h1>Motion QC</h1>"
            f"<p>Run-level motion summaries from fMRIPrep confounds. FD thresholds: warning={thresholds['warning']:.3f} mm, severe={thresholds['severe']:.3f} mm.</p>"
            + "<h2>Study Summary</h2>" + _summary_html(summary)
            + "<h2>Histograms</h2>" + "".join(f"<img class='histogram' src='figures/{html.escape(path)}' alt='Motion histogram'>" for path in histogram_files)
            + "<h2>Session / Task Summary</h2>" + _html_table(session_task, ["session", "task", "n_runs", "mean_mean_fd", "median_mean_fd", "mean_pct_fd_02", "mean_pct_fd_05"])
            + "<h2>Ranked Runs</h2>" + _html_table(frame, ["subject", "session", "task", "run", "mean_fd", "median_fd", "max_fd", "pct_fd_02", "pct_fd_05"])
            + "<h2>Flagged Runs</h2><p>Flags are review recommendations only; no subjects are automatically excluded.</p>"
            + _html_table(flagged, ["subject", "session", "task", "run", "mean_fd", "pct_fd_02", "pct_fd_05", "flag_level"])
            + "<h2>Session / Task Sorted FD Figures</h2>" + cohort_sections
            + "<h2>Individual FD Traces</h2>" + "".join(trace_sections)
            + "</body></html>\n"
        )
        report_path.write_text(report_html, encoding="utf-8")

        elapsed = round(time.perf_counter() - started, 3)
        metadata = {
            "report_type": self.name,
            "report_version": "2",
            "scope": self.scope,
            "status": "completed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "configuration": report_config,
            "fd_thresholds": thresholds,
            "number_of_runs_analyzed": len(runs),
            "number_of_subjects_analyzed": int(frame["subject"].nunique()),
            "source_files": source_files,
            "summary": summary,
            "warnings": warnings,
            "software": {
                "hbicproc": _package_version("hbicproc-pipeline"),
                "pandas": _package_version("pandas"),
                "matplotlib": _package_version("matplotlib"),
            },
            "outputs": [
                str(report_path),
                *[str(figure_dir / path) for path in histogram_files],
                *[str(output_dir / relative_path) for _, relative_path in trace_files],
                *[str(output_dir / relative_path) for relative_path in cohort_files.values()],
            ],
            "elapsed_seconds": elapsed,
        }
        write_json(output_dir / "metadata.json", metadata)
        logger.info("Generated motion_qc HTML and provenance in %.2f seconds", elapsed)
        return ReportResult(
            self.name,
            "completed",
            f"motion_qc report generated for {summary['total subjects']} subjects and {summary['total runs']} runs: {report_path}",
            output_dir,
            {"summary": summary, "warnings": warnings, "source_files": source_files},
        )
