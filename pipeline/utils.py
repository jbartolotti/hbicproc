import json
import subprocess
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default if default is not None else {}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def run_command(command, dry_run=False):
    if dry_run:
        return {
            "success": True,
            "message": "Dry run: command not executed.",
            "command": " ".join(str(part) for part in command),
        }

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"success": False, "message": f"Command not found: {exc}", "stderr": ""}

    if result.returncode != 0:
        return {
            "success": False,
            "message": f"Command failed with exit code {result.returncode}",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(str(part) for part in command),
        }

    return {
        "success": True,
        "message": "Command completed successfully.",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join(str(part) for part in command),
    }


def get_bids_root(config):
    if config.get("bids_root"):
        return Path(config["bids_root"])

    study_root = Path(config["study_root"])
    if study_root.exists():
        if any(p.is_dir() and p.name.startswith("sub-") for p in study_root.iterdir()):
            return study_root

    return Path(config["bidskit"]["output_dir"])


def list_subjects(config):
    bids_root_dir = get_bids_root(config)
    if not bids_root_dir.exists():
        return []
    subjects = [p.name for p in sorted(bids_root_dir.iterdir()) if p.is_dir() and p.name.startswith("sub-")]
    return subjects


def subject_dir(config, subject):
    return Path(config["study_root"]) / subject


def ensure_subject_directory(config, subject):
    subject_path = subject_dir(config, subject)
    subject_path.mkdir(parents=True, exist_ok=True)
    return subject_path


def format_next_command(cmd):
    return f"Next step:\n  {cmd}\n"
