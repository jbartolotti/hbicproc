from .base import BaseStage, StageResult
from ..processing.qc.service import run_reports as qc_report_run


class QCReportStage(BaseStage):
    name = "qc_report"
    state_key = None

    def run(self, subject, config, state, dry_run=False, rerun=False):
        _ = (subject, state)
        result = qc_report_run(config, dry_run=dry_run, rerun=rerun)
        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            details=result.get("details", {}),
        )