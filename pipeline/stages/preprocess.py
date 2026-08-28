from .base import BaseStage, StageResult
from ..processing.preprocess.service import run as preprocess_run


class PreprocessStage(BaseStage):
    name = "preprocess"
    state_key = "preprocessed"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        result = preprocess_run(subject, config, dry_run=dry_run, rerun=rerun)

        reserved = ("success", "skipped", "message", "next_command")
        details = {key: value for key, value in result.items() if key not in reserved}

        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            next_command=result.get("next_command"),
            details=details,
        )
