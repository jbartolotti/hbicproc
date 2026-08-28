from .base import BaseStage, StageResult
from ..processing.behavior.service import run as behavior_run


class BehaviorStage(BaseStage):
    name = "behavior"
    state_key = "behavior_complete"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        result = behavior_run(subject, config, dry_run=dry_run, rerun=rerun)

        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            next_command=result.get("next_command"),
            details=result.get("details", {}),
        )
