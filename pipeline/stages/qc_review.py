from pathlib import Path

from .base import BaseStage, StageResult
from ..utils import load_json


class QcReviewStage(BaseStage):
    name = "qc_review"
    state_key = "qc_reviewed"
    human_step = True

    def run(self, subject, config, state, dry_run=False, rerun=False):
        if state.get(self.state_key):
            return StageResult(success=True, message="QC review already recorded.")

        output_dir = Path(config["mriqc"]["output_dir"]) / subject
        if not output_dir.exists():
            return StageResult(
                success=False,
                message=(
                    f"MRIQC output not found at {output_dir}. \n"
                    f"Run: hbicproc qc {subject}"
                ),
            )

        report_files = sorted(output_dir.rglob("*.html"))
        json_files = sorted(output_dir.rglob("*.json"))

        if not report_files:
            return StageResult(
                success=False,
                message=(
                    f"No MRIQC HTML reports were found under {output_dir}.\n"
                    f"Run: hbicproc qc {subject}"
                ),
            )

        html_report = report_files[0]
        summary = ""
        summary_json = next((p for p in json_files if p.name.startswith(subject) and p.suffix == ".json"), None)
        if summary_json:
            qc_data = load_json(summary_json, default={})
            if isinstance(qc_data, dict):
                summary_keys = sorted(qc_data.keys())[:5]
                if summary_keys:
                    summary = f"Found summary JSON {summary_json.name}. Top keys: {', '.join(summary_keys)}"
        else:
            summary = f"Found {len(json_files)} JSON files in the MRIQC output."

        exclusions_file = Path(config["hbicproc"]["exclusions_file"])

        return StageResult(
            success=False,
            message=(
                f"Manual QC review is required for {subject}.\n"
                f"Open the MRIQC HTML report: {html_report}\n"
                f"{summary}\n\n"
                "Record run exclusions with:\n"
                f"  hbicproc exclude {subject} --run <task-run-label>\n"
                "If all runs are acceptable, mark QC review complete with:\n"
                f"  hbicproc exclude {subject} --clear\n\n"
                f"Exclusions are stored in: {exclusions_file}\n"
                "QC review will remain incomplete until you run `hbicproc exclude`."
            ),
            needs_review=True,
            next_command=f"hbicproc exclude {subject} --run <task-run-label>",
            details={
                "report": str(html_report),
                "report_count": len(report_files),
                "exclusions_file": str(exclusions_file),
            },
        )
