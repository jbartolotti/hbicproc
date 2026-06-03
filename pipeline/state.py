import json
from pathlib import Path

STATE_FILE_NAME = ".pipeline_state.json"
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


def load_subject_state(subject_dir):
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


def save_subject_state(subject_dir, state):
    state_path = state_file(subject_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
