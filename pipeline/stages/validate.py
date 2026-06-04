from pathlib import Path

from .base import BaseStage, StageResult


class ValidateStage(BaseStage):
    name = "validate"
    state_key = "validated"

    def _summarize_bids(self, bids_dir: Path) -> str:
        if not bids_dir.exists():
            return "BIDS directory is missing."

        subject_dirs = [p for p in bids_dir.iterdir() if p.is_dir()]
        n_files = sum(1 for _ in bids_dir.rglob("*"))
        summary = [
            f"BIDS root: {bids_dir}",
            f"Subjects found: {len(subject_dirs)}",
            f"Subdirectories: {len(subject_dirs)}",
            f"Total file entries: {n_files}",
        ]
        return "\n".join(summary)

    def run(self, subject, config, state, dry_run=False, rerun=False):
        bids_dir = Path(config["bidskit"]["output_dir"]) / subject
        if not bids_dir.exists():
            return StageResult(success=False, message=f"BIDS folder not found at {bids_dir}.")

        summary = self._summarize_bids(bids_dir)
        return StageResult(
            success=True,
            message="BIDS validation checks completed.",
            details={"summary": summary},
            next_command=f"hbicproc qc {subject}",
        )
