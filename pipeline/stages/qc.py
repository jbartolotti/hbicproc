from pathlib import Path
import shutil

from .base import BaseStage, StageResult
from ..container import SingularityRunner
from ..utils import ensure_dir, get_bids_root


class QcStage(BaseStage):
    name = "qc"
    state_key = "qc_complete"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        bids_dir = get_bids_root(config) / subject
        if not bids_dir.exists():
            return StageResult(success=False, message=f"BIDS source directory not found: {bids_dir}")

        image = config["mriqc"]["singularity_image"]
        if not any(image.startswith(prefix) for prefix in ("docker://", "shub://", "library://")):
            if not Path(image).exists():
                return StageResult(success=False, message=f"Singularity image not found: {image}")

        output_dir = Path(config["mriqc"]["output_dir"]) / subject
        work_dir = Path(config["mriqc"]["work_dir"]) / subject

        if rerun:
            if output_dir.exists() and not dry_run:
                shutil.rmtree(output_dir)
            if work_dir.exists() and not dry_run:
                shutil.rmtree(work_dir)

        ensure_dir(output_dir)
        ensure_dir(work_dir)

        extra_args = config["mriqc"].get("extra_args", "participant --participant_label {subject}").format(subject=subject)
        runner = SingularityRunner(image)
        result = runner.run(
            input_dir=bids_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            extra_args=extra_args,
            dry_run=dry_run,
        )

        if not result["success"]:
            return StageResult(success=False, message=result.get("message", "MRIQC failed."), details=result)

        report_files = sorted(output_dir.rglob("*.html"))
        if not report_files:
            return StageResult(
                success=False,
                message=(
                    f"MRIQC completed but no HTML reports were found under {output_dir}.\n"
                    "Check the MRIQC output directory and container logs."
                ),
                details={**result, "output_dir": str(output_dir)},
            )

        report_list = [str(path) for path in report_files[:5]]
        message = (
            f"MRIQC participant stage completed for {subject}.\n"
            f"Review HTML reports in: {output_dir}\n"
            f"Most likely report: {report_list[0]}\n\n"
            "Then run: hbicproc qc_review {subject}"
        )

        return StageResult(
            success=True,
            message=message,
            next_command=f"hbicproc qc_review {subject}",
            details={
                "command": result.get("command"),
                "output_dir": str(output_dir),
                "reports": report_list,
            },
        )
