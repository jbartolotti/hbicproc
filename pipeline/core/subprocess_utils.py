import subprocess
import sys
from pathlib import Path


def run_command(command, dry_run=False):
    command_str = " ".join(str(part) for part in command)
    if dry_run:
        print(f"Dry run command: {command_str}")
        return {
            "success": True,
            "message": "Dry run: command not executed.",
            "command": command_str,
        }

    print(f"Executing command: {command_str}")
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        return {"success": False, "message": f"Command not found: {exc}", "stderr": "", "command": command_str}

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        return {
            "success": False,
            "message": f"Command failed with exit code {result.returncode}",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": command_str,
        }

    return {
        "success": True,
        "message": "Command completed successfully.",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command_str,
    }


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
