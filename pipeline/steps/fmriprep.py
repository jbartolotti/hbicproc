from pathlib import Path
from ._utils import path_exists, run_subprocess


def run(subject, config, dry_run=False):
    output_dir = Path(config["fmriprep"]["output_dir"]) / subject
    if path_exists(output_dir):
        return {
            "success": True,
            "skipped": True,
            "message": f"fMRIPrep output already exists at {output_dir}",
            "command": "",
            "returncode": None,
        }

    command = [
        "singularity",
        "run",
        config["fmriprep"]["singularity_image"],
        config["bidskit"]["output_dir"],
        str(output_dir),
        "--fs-license-file",
        config["fmriprep"].get("fs_license_file", ""),
    ]
    extra = config["fmriprep"].get("extra_args", "").strip().format(subject=subject)
    if extra:
        command.extend(extra.split())

    status = run_subprocess(command, dry_run=dry_run)
    if status["success"] and not dry_run:
        status["message"] = f"fMRIPrep launched for {subject}."
    return status
