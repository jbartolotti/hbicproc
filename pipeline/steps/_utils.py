import subprocess
import sys
from pathlib import Path


def run_subprocess(command, dry_run=False):
    full_command = [str(arg) for arg in command]
    if dry_run:
        return {
            "success": True,
            "skipped": False,
            "message": "Dry-run only; command not executed.",
            "command": " ".join(full_command),
            "returncode": None,
        }

    try:
        print("Executing:", " ".join(full_command))
        result = subprocess.run(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print(result.stderr, end="", file=sys.stderr)

        return {
            "success": result.returncode == 0,
            "skipped": False,
            "message": "Command completed." if result.returncode == 0 else "Command failed.",
            "command": " ".join(full_command),
            "returncode": result.returncode,
        }
    except FileNotFoundError as exc:
        return {
            "success": False,
            "skipped": False,
            "message": f"Executable not found: {exc}",
            "command": " ".join(full_command),
            "returncode": None,
        }


def path_exists(path):
    return Path(path).exists()
