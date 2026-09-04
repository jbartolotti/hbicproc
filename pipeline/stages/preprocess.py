from .base import BaseStage, StageResult
from ..processing.preprocess.service import run as preprocess_run
from ..core.paths import get_bids_root
from ..state import invalidate_bids_index


class PreprocessStage(BaseStage):
    name = "preprocess"
    state_key = "preprocessed"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        result = preprocess_run(subject, config, dry_run=dry_run, rerun=rerun)
        if result.get("success", False) and not dry_run:
            invalidate_bids_index("fmriprep", get_bids_root(config))

        reserved = ("success", "skipped", "message", "next_command")
        details = {key: value for key, value in result.items() if key not in reserved}

        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            next_command=result.get("next_command"),
            details=details,
        )
