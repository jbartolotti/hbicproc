from .base import BaseStage, StageResult


class QcReviewStage(BaseStage):
    name = "qc_review"
    state_key = "qc_reviewed"
    human_step = True

    def run(self, subject, config, state, dry_run=False):
        if state.get(self.state_key):
            return StageResult(success=True, message="QC review already recorded.")

        return StageResult(
            success=False,
            message=(
                "Manual QC review is required before preprocessing.\n"
                f"Mark exclusions for {subject}: hbicproc exclude {subject} --run <task-run-label>"
            ),
            needs_review=True,
        )
