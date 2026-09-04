import json
from pathlib import Path

from .core.paths import get_bids_root

SUMMARY_FILE_NAME = "pipeline_summary.json"
PIPELINE_STATE_FILE_NAME = "pipeline_state.json"
PYBIDS_CACHE_DIR = Path("code") / "cache" / "pybids"
DEFAULT_STATE = {
    "downloaded": False,
    "bidsified": False,
    "validated": False,
    "qc_complete": False,
    "qc_reviewed": False,
    "preprocessed": False,
}


def pipeline_state_file(study_root="."):
    return Path(study_root) / "code" / "cache" / PIPELINE_STATE_FILE_NAME


def load_pipeline_state(study_root="."):
    path = pipeline_state_file(study_root)
    if not path.exists():
        return {"bids_indexes": {}}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {"bids_indexes": {}}

    if not isinstance(data, dict):
        return {"bids_indexes": {}}
    indexes = data.get("bids_indexes")
    if not isinstance(indexes, dict):
        data["bids_indexes"] = {}
    return data


def save_pipeline_state(study_root, state):
    path = pipeline_state_file(study_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")


def invalidate_bids_index(index_name, study_root="."):
    index_name = str(index_name).strip()
    if not index_name:
        raise ValueError("index_name must not be empty.")

    state = load_pipeline_state(study_root)
    indexes = state.setdefault("bids_indexes", {})
    current_revision = indexes.get(index_name, 0)
    try:
        current_revision = int(current_revision)
    except (TypeError, ValueError):
        current_revision = 0
    indexes[index_name] = current_revision + 1
    save_pipeline_state(study_root, state)
    return indexes[index_name]


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
