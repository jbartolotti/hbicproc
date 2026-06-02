from pathlib import Path
from ._utils import path_exists, run_subprocess


def run(subject, config, dry_run=False):
    bids_dir = Path(config["bidskit"]["output_dir"]) / subject
    if path_exists(bids_dir):
        return {
            "success": True,
            "skipped": True,
            "message": f"BIDS output already exists at {bids_dir}",
            "command": "",
            "returncode": None,
        }

    command = [
        "bidskit",
        "--input-dir",
        config["bidskit"]["input_dir"],
        "--output-dir",
        config["bidskit"]["output_dir"],
        "--subject",
        subject,
    ]
    extra = config["bidskit"].get("extra_args", "").strip()
    if extra:
        command.extend(extra.split())

    status = run_subprocess(command, dry_run=dry_run)
    if status["success"] and not dry_run:
        status["message"] = f"BIDS conversion launched for {subject}."
    return status
