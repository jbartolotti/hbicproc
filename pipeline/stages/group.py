from .base import BaseStage, StageResult
from ..processing.group.service import run as group_run


class GroupStage(BaseStage):
    name = "group"
    state_key = None

    def run(self, subject, config, state, dry_run=False, rerun=False):
        _ = (subject, state, rerun)
        result = group_run(config, dry_run=dry_run)
        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            details=result.get("details", {}),
        )
