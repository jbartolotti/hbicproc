import json
from pathlib import Path

from .core.paths import get_bids_root

SUMMARY_FILE_NAME = "pipeline_summary.json"
DEFAULT_STATE = {
    "downloaded": False,
    "bidsified": False,
    "validated": False,
    "qc_complete": False,
    "qc_reviewed": False,
    "preprocessed": False,
}


def summary_file(config):
    bids_root = get_bids_root(config)
    return Path(bids_root) / "code" / SUMMARY_FILE_NAME


def load_pipeline_summary(config):
    path = summary_file(config)
    if not path.exists():
        return {"subjects": {}}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {"subjects": {}}

    if not isinstance(data, dict):
        return {"subjects": {}}

    data.setdefault("subjects", {})
    return data


def save_pipeline_summary(config, summary):
    path = summary_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def load_subject_state(config, subject):
    summary = load_pipeline_summary(config)
    subject_entry = summary.get("subjects", {}).get(subject, {})
    state = DEFAULT_STATE.copy()
    for key in DEFAULT_STATE:
        if key in subject_entry:
            state[key] = subject_entry[key]
    return state


def save_subject_state(config, subject, state):
    summary = load_pipeline_summary(config)
    subjects = summary.setdefault("subjects", {})
    subject_entry = subjects.setdefault(subject, {})
    for key in DEFAULT_STATE:
        if key in state:
            subject_entry[key] = state[key]
    subjects[subject] = subject_entry
    summary["subjects"] = subjects
    save_pipeline_summary(config, summary)


def update_session_state(config, subject, session, updates):
    summary = load_pipeline_summary(config)
    subjects = summary.setdefault("subjects", {})
    subject_entry = subjects.setdefault(subject, {})
    sessions = subject_entry.setdefault("sessions", {})
    session_entry = sessions.setdefault(session, {})
    session_entry.update(updates)
    sessions[session] = session_entry
    subject_entry["sessions"] = sessions
    subjects[subject] = subject_entry
    summary["subjects"] = subjects
    save_pipeline_summary(config, summary)
