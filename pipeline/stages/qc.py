from .base import BaseStage, StageResult
from ..processing.qc.service import run as qc_run


class QcStage(BaseStage):
    name = "qc"
    state_key = "qc_complete"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        result = qc_run(subject, config, dry_run=dry_run, rerun=rerun)

        return StageResult(
            success=result["success"],
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            next_command=result.get("next_command"),
            details=result.get("details", {}),
        )

