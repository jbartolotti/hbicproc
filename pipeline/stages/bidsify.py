from pathlib import Path

from .base import BaseStage, StageResult
from ..utils import ensure_dir, run_command


class BidsifyStage(BaseStage):
    name = "bidsify"
    state_key = "bidsified"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        input_dir = Path(config["bidskit"]["input_dir"]) / subject
        output_dir = Path(config["bidskit"]["output_dir"]) / subject
        ensure_dir(output_dir)

        if not input_dir.exists():
            return StageResult(success=False, message=f"Input data for {subject} not found at {input_dir}.")

        extra_args = config["bidskit"].get("extra_args", "").strip()
        if extra_args:
            command = [
                "bidskit",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--participant-label",
                subject,
            ]
            command.extend(extra_args.split())
            result = run_command(command, dry_run=dry_run)
            return StageResult(
                success=result["success"],
                message=result.get("message", "BIDS conversion completed."),
                details={"command": result.get("command"), "stdout": result.get("stdout"), "stderr": result.get("stderr")},
            )

        return StageResult(
            success=True,
            message=(
                f"BIDS output prepared at {output_dir}."
                " If you want a real conversion, set `bidskit.extra_args` in config." 
            ),
            details={"output_dir": str(output_dir)},
        )
