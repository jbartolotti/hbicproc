from .base import BaseStage, StageResult
from ..processing.analysis.service import run as analysis_run


class AnalysisStage(BaseStage):
    name = "analysis"
    # Analysis completion is determined per run from its core output products.
    state_key = None

    def run(self, subject, config, state, dry_run=False, rerun=False, plugin_name=None):
        result = analysis_run(subject, config, dry_run=dry_run, rerun=rerun, plugin_name=plugin_name)

        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            next_command=result.get("next_command"),
            details=result.get("details", {}),
        )
