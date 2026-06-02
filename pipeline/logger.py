import json
from datetime import datetime
from pathlib import Path


def get_log_paths(config, subject=None):
    log_dir = Path(config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    if subject:
        json_path = log_dir / f"pipeline_{subject}.jsonl"
        text_path = log_dir / f"pipeline_{subject}.log"
    else:
        json_path = log_dir / "pipeline.jsonl"
        text_path = log_dir / "pipeline.log"
    return json_path, text_path


def append_log(subject, step, status, config):
    json_path, text_path = get_log_paths(config, subject)
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
    with json_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    append_text_log(entry, text_path)
    return json_path


def append_text_log(entry, text_path):
    timestamp = entry["timestamp"]
    subject = entry.get("subject") or "global"
    step = entry.get("step") or "unknown"
    status = "SKIPPED" if entry.get("skipped") else "SUCCESS" if entry.get("success") else "FAILED"
    command = entry.get("command", "")
    message = entry.get("message", "")
    returncode = entry.get("returncode")

    line = (
        f"[{timestamp}] subject={subject} step={step} status={status} returncode={returncode}"
        f" message={message} command={command}"
    )
    with Path(text_path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return text_path


def append_event(message, config, subject=None, step=None):
    _, text_path = get_log_paths(config, subject)
    timestamp = datetime.utcnow().isoformat() + "Z"
    subject_part = f"subject={subject}" if subject else "subject=global"
    step_part = f"step={step}" if step else "step=event"
    line = f"[{timestamp}] {subject_part} {step_part} message={message}"
    with Path(text_path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return text_path
