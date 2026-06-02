from pathlib import Path
from ._utils import path_exists, run_subprocess


def run(subject, config, dry_run=False):
    output_dir = Path(config["xnat"]["output_dir"]) / subject
    if path_exists(output_dir):
        return {
            "success": True,
            "skipped": True,
            "message": f"Download output already exists at {output_dir}",
            "command": "",
            "returncode": None,
        }

    command = [
        "Rscript",
        config["xnat"]["script_path"],
        "--subject",
        subject,
        "--output-dir",
        str(output_dir),
    ]

    status = run_subprocess(command, dry_run=dry_run)
    if status["success"] and not dry_run:
        status["message"] = f"XNAT download launched for {subject}."
    return status
