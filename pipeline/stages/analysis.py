from .base import BaseStage, StageResult
from ..processing.analysis.service import run as analysis_run


class AnalysisStage(BaseStage):
    name = "analysis"
    # Completion will be tracked at the model/analysis derivative level.
    state_key = None

    def run(self, subject, config, state, dry_run=False, rerun=False):
        _ = state
        result = analysis_run(
            subject,
            config,
            dry_run=dry_run,
            rerun=rerun,
        )

        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            next_command=result.get("next_command"),
            details=result.get("details", {}),
        )
