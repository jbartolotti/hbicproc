import json
from datetime import datetime
from pathlib import Path


def get_log_path(subject, config):
    log_dir = Path(config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"pipeline_{subject}.jsonl"


def append_log(subject, step, status, config):
    log_path = get_log_path(subject, config)
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "subject": subject,
        "step": step,
        "success": status.get("success", False),
        "skipped": status.get("skipped", False),
        "message": status.get("message", ""),
        "command": status.get("command", ""),
        "returncode": status.get("returncode")
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return log_path
