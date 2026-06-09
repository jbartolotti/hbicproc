from pathlib import Path

from .base import BaseStage, StageResult
from ..steps.download import run as download_run


class DownloadStage(BaseStage):
    name = "download"
    state_key = "downloaded"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        output_dir = Path(config["xnat"]["output_dir"])
        result = download_run(subject, config, dry_run=dry_run, rerun=rerun)

        return StageResult(
            success=result["success"],
            skipped=result.get("skipped", False),
            message=result.get("message", "Download completed."),
            details={
                "output_dir": str(output_dir / subject),
                "command": result.get("command", ""),
            },
        )
