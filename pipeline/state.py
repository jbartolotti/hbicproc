import json
from pathlib import Path

from .utils import get_bids_root

STATE_FILE_NAME = ".pipeline_state.json"
SUMMARY_FILE_NAME = "pipeline_summary.json"
DEFAULT_STATE = {
    "downloaded": False,
    "bidsified": False,
    "validated": False,
    "qc_complete": False,
    "qc_reviewed": False,
    "preprocessed": False,
}


def state_file(subject_dir):
    return Path(subject_dir) / STATE_FILE_NAME


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


def load_subject_state(subject_dir, config=None):
    if config is not None:
        summary = load_pipeline_summary(config)
        subject_key = Path(subject_dir).name
        subject_entry = summary.get("subjects", {}).get(subject_key, {})
        state = DEFAULT_STATE.copy()
        for key in DEFAULT_STATE:
            if key in subject_entry:
                state[key] = subject_entry[key]
        return state

    state_path = state_file(subject_dir)
    if not state_path.exists():
        return DEFAULT_STATE.copy()
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    state = DEFAULT_STATE.copy()
    state.update(data)
    return state


def save_subject_state(subject_dir, state, config=None):
    if config is not None:
        summary = load_pipeline_summary(config)
        subjects = summary.setdefault("subjects", {})
        subject_key = Path(subject_dir).name
        subject_entry = subjects.setdefault(subject_key, {})
        for key in DEFAULT_STATE:
            if key in state:
                subject_entry[key] = state[key]
        subjects[subject_key] = subject_entry
        summary["subjects"] = subjects
        save_pipeline_summary(config, summary)
        return

    state_path = state_file(subject_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")


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
