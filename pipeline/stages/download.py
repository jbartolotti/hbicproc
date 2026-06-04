from pathlib import Path

from .base import BaseStage, StageResult
from ..utils import ensure_dir, run_command


class DownloadStage(BaseStage):
    name = "download"
    state_key = "downloaded"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        xnat_script = Path(config["xnat"]["script_path"])
        output_dir = Path(config["xnat"]["output_dir"]) / subject
        ensure_dir(output_dir)

        if not xnat_script.exists():
            return StageResult(
                success=True,
                message=(
                    f"Download stub created at {output_dir}."
                    " Add XNAT logic in config['xnat']['script_path'] to perform real download."
                ),
                details={"subject_dir": str(output_dir)},
            )

        command = ["Rscript", str(xnat_script), subject, str(output_dir)]
        result = run_command(command, dry_run=dry_run)

        return StageResult(
            success=result["success"],
            message=result.get("message", "Download completed."),
            details={
                "command": result.get("command"),
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
                "output_dir": str(output_dir),
            },
        )
