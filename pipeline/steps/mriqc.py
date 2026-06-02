from pathlib import Path
from ._utils import path_exists, run_subprocess


def run(subject, config, dry_run=False):
    output_dir = Path(config["mriqc"]["output_dir"]) / subject
    if path_exists(output_dir):
        return {
            "success": True,
            "skipped": True,
            "message": f"MRIQC output already exists at {output_dir}",
            "command": "",
            "returncode": None,
        }

    command = [
        "singularity",
        "run",
        config["mriqc"]["singularity_image"],
        config["bidskit"]["output_dir"],
        str(output_dir),
    ]
    extra = config["mriqc"].get("extra_args", "").strip().format(subject=subject)
    if extra:
        command.extend(extra.split())

    status = run_subprocess(command, dry_run=dry_run)
    if status["success"] and not dry_run:
        status["message"] = f"MRIQC launched for {subject}."
    return status
