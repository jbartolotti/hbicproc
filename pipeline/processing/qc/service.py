from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...core.paths import write_json
from ...logger import append_event
from ..analysis.dataset import DatasetIndex
from .base import ReportResult
from .registry import get_report_classes

logger = logging.getLogger(__name__)


@dataclass
class QCReportContext:
    config: dict[str, Any]
    output_dir: Path
    report_name: str
    report_config: dict[str, Any]
    dataset_index: DatasetIndex


def _normalized_reports(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, list):
        return {str(name): {"enabled": True} for name in value}
    if isinstance(value, dict):
        return {str(name): dict(settings) for name, settings in value.items()}
    return {}


def _write_index(output_dir: Path, results: list[ReportResult]) -> None:
    rows = []
    for result in results:
        report_link = f"{result.name}/report.html" if result.status == "completed" else ""
        link = f"<a href='{html.escape(report_link)}'>open report</a>" if report_link else ""
        rows.append(
            f"<tr><td>{html.escape(result.name)}</td><td>{html.escape(result.status)}</td>"
            f"<td>{link}</td><td>{html.escape(result.message)}</td></tr>"
        )
    (output_dir / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>QC reports</title></head><body>"
        "<h1>QC reports</h1><table><thead><tr><th>Report</th><th>Status</th>"
        f"<th>Link</th><th>Message</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</body></html>\n",
        encoding="utf-8",
    )


def run_reports(config: dict[str, Any], *, dry_run: bool = False, rerun: bool = False) -> dict[str, Any]:
    qc_config = config.get("qc_report", {})
    if not qc_config.get("enabled", True):
        return {"success": True, "skipped": True, "message": "QC reporting is disabled."}

    output_dir = Path(qc_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {"success": True, "message": "QC report dry run completed.", "details": {"output_dir": str(output_dir)}}

    reports = _normalized_reports(qc_config.get("reports", {}))
    report_classes = get_report_classes()
    try:
        dataset_index = DatasetIndex.from_config(config)
    except Exception as exc:
        return {"success": False, "message": f"Could not initialize QC dataset index: {exc}"}

    results: list[ReportResult] = []
    for report_name, report_config in reports.items():
        if not report_config.get("enabled", True):
            continue
        report_class = report_classes.get(report_name)
        if report_class is None:
            logger.warning("QC report '%s' is not registered; skipping.", report_name)
            results.append(ReportResult(report_name, "skipped", f"Unknown QC report '{report_name}'."))
            continue
        context = QCReportContext(config, output_dir, report_name, report_config, dataset_index)
        append_event(f"QC report started: {report_name}", config, step="qc_report")
        try:
            result = report_class().generate(context)
        except Exception as exc:
            logger.exception("QC report '%s' failed", report_name)
            result = ReportResult(report_name, "failed", f"QC report failed: {exc}")
        results.append(result)
        append_event(f"QC report {result.status}: {report_name}", config, step="qc_report")

    _write_index(output_dir, results)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": [
            {"name": result.name, "status": result.status, "message": result.message}
            for result in results
        ],
        "output_dir": str(output_dir),
        "rerun": rerun,
    }
    write_json(output_dir / "manifest.json", manifest)
    failed = [result for result in results if result.status == "failed"]
    skipped = [result for result in results if result.status == "skipped"]
    message = f"QC reporting completed: {len(results) - len(failed) - len(skipped)} generated, {len(skipped)} skipped, {len(failed)} failed."
    return {
        "success": not failed,
        "message": message,
        "details": {"manifest": str(output_dir / "manifest.json"), "reports": manifest["reports"]},
    }

from pathlib import Path
import shutil

from ...core.container import SingularityRunner
from ...core.paths import ensure_dir, get_bids_root


def run(subject, config, dry_run=False, rerun=False):
    bids_root = get_bids_root(config)
    bids_dir = bids_root / subject
    if not bids_dir.exists():
        return {"success": False, "message": f"BIDS subject directory not found: {bids_dir}"}

    image = config["mriqc"]["singularity_image"]
    if not any(image.startswith(prefix) for prefix in ("docker://", "shub://", "library://")):
        if not Path(image).exists():
            return {"success": False, "message": f"Singularity image not found: {image}"}

    output_dir = Path(config["mriqc"]["output_dir"])
    subject_output_dir = output_dir / subject
    base_work_dir = Path(config["mriqc"]["work_dir"])
    work_dir = base_work_dir / f"work_{subject}"

    if rerun and subject_output_dir.exists() and not dry_run:
        shutil.rmtree(subject_output_dir)

    ensure_dir(output_dir)
    ensure_dir(work_dir)

    internal_input_dir = Path("/bids_root")
    internal_work_dir = Path("/work")

    binds = [
        (str(bids_root.resolve()), str(internal_input_dir)),
        (str(work_dir.resolve()), str(internal_work_dir)),
    ]

    try:
        internal_output_dir = Path("/bids_root") / output_dir.relative_to(bids_root)
    except ValueError:
        internal_output_dir = Path("/output")
        binds.append((str(output_dir.resolve()), str(internal_output_dir)))

    extra_args = config["mriqc"].get("extra_args", "participant --participant_label {subject}").format(subject=subject)
    if "-w" not in extra_args and "--work-dir" not in extra_args:
        extra_args = f"{extra_args} -w {internal_work_dir}"

    print(f"QC stage for {subject}")
    print(f"  BIDS root: {bids_root}")
    print(f"  MRIQC output directory: {output_dir}")
    print(f"  MRIQC work directory: {work_dir}")
    print(f"  Binding BIDS root to container path {internal_input_dir}")
    print(f"  Binding work dir to container path {internal_work_dir}")

    runner = SingularityRunner(image, clean_env=True)
    result = runner.run(
        extra_args=(f"{internal_input_dir} {internal_output_dir} {extra_args}"),
        dry_run=dry_run,
        binds=binds,
        clean_env=True,
    )

    if not result["success"]:
        return {"success": False, "message": result.get("message", "MRIQC failed."), "details": result}

    report_files = sorted(subject_output_dir.rglob("*.html"))
    if not report_files:
        return {
            "success": False,
            "message": (
                f"MRIQC completed but no HTML reports were found under {output_dir}.\n"
                "Check the MRIQC output directory and container logs."
            ),
            "details": {**result, "output_dir": str(output_dir)},
        }

    report_list = [str(path) for path in report_files[:5]]
    message = (
        f"MRIQC participant stage completed for {subject}.\n"
        f"Review HTML reports in: {output_dir}\n"
        f"Most likely report: {report_list[0]}\n\n"
        "Then run: hbicproc qc_review {subject}"
    )

    return {
        "success": True,
        "message": message,
        "next_command": f"hbicproc qc_review {subject}",
        "details": {
            "command": result.get("command"),
            "output_dir": str(output_dir),
            "reports": report_list,
        },
    }
