from pathlib import Path

from .base import BaseStage, StageResult
from ..container import SingularityRunner
from ..utils import ensure_dir


class QcStage(BaseStage):
    name = "qc"
    state_key = "qc_complete"

    def run(self, subject, config, state, dry_run=False):
        bids_dir = Path(config["bidskit"]["output_dir"])
        if not bids_dir.exists():
            return StageResult(success=False, message=f"BIDS source directory not found: {bids_dir}")

        output_dir = Path(config["mriqc"]["output_dir"]) / subject
        work_dir = Path(config["mriqc"]["work_dir"]) / subject
        ensure_dir(output_dir)
        ensure_dir(work_dir)

        extra_args = config["mriqc"].get("extra_args", "participant --participant_label {subject}").format(subject=subject)
        runner = SingularityRunner(config["mriqc"]["singularity_image"])
        result = runner.run(
            input_dir=bids_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            extra_args=extra_args,
            dry_run=dry_run,
        )

        if not result["success"]:
            return StageResult(success=False, message=result.get("message", "MRIQC failed."), details=result)

        next_command = f"hbicproc exclude {subject} --run <task-run-label>"
        return StageResult(
            success=True,
            message=(
                f"MRIQC participant stage completed for {subject}.\n"
                f"Review HTML reports in: {output_dir}\n\n"
                "Then mark run exclusions and continue with preprocessing."
            ),
            next_command=next_command,
            details={"command": result.get("command"), "output_dir": str(output_dir)},
        )
