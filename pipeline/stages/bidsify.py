from pathlib import Path

from .base import BaseStage, StageResult
from ..processing.bidsify.service import run as bidsify_run


class BidsifyStage(BaseStage):
    name = "bidsify"
    state_key = "bidsified"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        input_dir = Path(config["bidskit"]["input_dir"]) / subject
        output_dir = Path(config["bidskit"]["output_dir"]) / subject

        if not input_dir.exists():
            return StageResult(success=False, message=f"Input data for {subject} not found at {input_dir}.")

        result = bidsify_run(subject, config, dry_run=dry_run)

        return StageResult(
            success=result["success"],
            skipped=result.get("skipped", False),
            message=result.get("message", "BIDS conversion completed."),
            details={
                "output_dir": str(output_dir),
                "command": result.get("command", ""),
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
            },
        )

