from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from ...core.paths import write_json
from ..analysis.context import InputDataset
from .base import QCReport, ReportResult
from .registry import register_report

logger = logging.getLogger(__name__)
_SUBJECT_PATTERN = re.compile(r"(?:^|[\\/])sub-([^_\\/]+)")


def _subject_from_path(path: Path) -> str | None:
    match = _SUBJECT_PATTERN.search(str(path))
    return f"sub-{match.group(1)}" if match else None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@register_report
class MotionQCReport(QCReport):
    name = "motion_qc"
    scope = "subject"

    def generate(self, context) -> ReportResult:
        output_dir = context.output_dir / self.name
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        report_config = context.report_config
        input_dataset_config = report_config.get("input_dataset") or context.config["analysis"]["input_dataset"]
        dataset = InputDataset(
            str(input_dataset_config["name"]),
            Path(input_dataset_config["path"]),
        )
        confounds_files = context.dataset_index.get_confounds_files(input_dataset=dataset)
        values: dict[str, list[float]] = {}
        source_files: list[str] = []
        warnings: list[str] = []

        for path in confounds_files:
            subject = _subject_from_path(path)
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
            displacement = pd.to_numeric(confounds["framewise_displacement"], errors="coerce").dropna()
            if displacement.empty:
                warnings.append(f"No numeric framewise displacement values in {path}")
                continue
            values.setdefault(subject, []).extend(displacement.tolist())
            source_files.append(str(path))

        if not values:
            message = "motion_qc skipped: no usable fMRIPrep confounds files were found."
            return ReportResult(self.name, "skipped", message, output_dir, {"warnings": warnings})

        means = {subject: sum(subject_values) / len(subject_values) for subject, subject_values in values.items()}
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        subjects = sorted(means)
        figure_path = figure_dir / "mean_framewise_displacement.png"
        figure, axis = plt.subplots(figsize=(max(6, len(subjects) * 0.35), 4.5))
        axis.bar(subjects, [means[subject] for subject in subjects])
        axis.set_xlabel("Subject")
        axis.set_ylabel("Mean framewise displacement (mm)")
        axis.set_title("Mean framewise displacement by subject")
        axis.tick_params(axis="x", rotation=90)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=150)
        plt.close(figure)

        table_rows = "\n".join(
            f"<tr><td>{html.escape(subject)}</td><td>{means[subject]:.6f}</td></tr>"
            for subject in subjects
        )
        report_path = output_dir / "report.html"
        report_path.write_text(
            "<!doctype html>\n"
            "<html><head><meta charset='utf-8'><title>Motion QC</title></head><body>\n"
            "<h1>Motion QC</h1>\n"
            "<p>Mean framewise displacement calculated from usable fMRIPrep confounds values.</p>\n"
            "<img src='figures/mean_framewise_displacement.png' alt='Mean framewise displacement by subject'>\n"
            "<table><thead><tr><th>Subject</th><th>Mean FD (mm)</th></tr></thead>"
            f"<tbody>{table_rows}</tbody></table>\n"
            "</body></html>\n",
            encoding="utf-8",
        )

        metadata = {
            "report_type": self.name,
            "report_version": "1",
            "scope": self.scope,
            "status": "completed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "configuration": report_config,
            "source_files": source_files,
            "subjects": subjects,
            "mean_framewise_displacement": means,
            "warnings": warnings,
            "software": {
                "hbicproc": _package_version("hbicproc-pipeline"),
                "pandas": _package_version("pandas"),
                "matplotlib": _package_version("matplotlib"),
            },
            "outputs": [str(report_path), str(figure_path)],
        }
        write_json(output_dir / "metadata.json", metadata)
        logger.info("Generated motion_qc report for %d subjects at %s", len(subjects), report_path)
        return ReportResult(
            self.name,
            "completed",
            f"motion_qc report generated for {len(subjects)} subjects: {report_path}",
            output_dir,
            {"subjects": subjects, "source_files": source_files, "warnings": warnings},
        )