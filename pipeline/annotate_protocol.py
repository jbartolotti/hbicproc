import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from prompt_toolkit.shortcuts import button_dialog, input_dialog, radiolist_dialog
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

BIDS_DIR_OPTIONS = ["anat", "func", "fmap", "dwi", "unknown"]
DEFAULT_STYLE = Style.from_dict({
    "dialog": "bg:#202020",
    "button": "bg:#008800 #ffffff",
    "button.focused": "bg:#00ff00 #000000",
}) if PROMPT_TOOLKIT_AVAILABLE else None


def load_protocol(file_path: str) -> Dict[str, List[str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Protocol file does not exist: {file_path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_sequences(data: Dict[str, List[str]]) -> List[str]:
    sequence_names = list(data.keys())
    print("\nSequences found in protocol file:")
    for index, name in enumerate(sequence_names, start=1):
        print(f"  {index}. {name}")
    print(f"\nTotal sequences: {len(sequence_names)}\n")
    return sequence_names


def suggest_labels(sequence_name: str, all_sequence_names: List[str]) -> Dict[str, str]:
    lower = sequence_name.lower()
    hints = {
        "bids_dir": "unknown",
        "bids_name": "",
        "intended_for": "",
    }

    anat_tokens = ["mprage", "t1", "t2", "anat", "mpr", "sag", "t1w", "t2w"]
    func_tokens = ["bold", "fmri", "rest", "nback", "flanker", "task", "functional"]
    fmap_tokens = ["fieldmap", "field_map", "fmap", "phase", "phasediff", "magnitude", "ap", "pa"]
    dwi_tokens = ["dwi", "diff", "dtifit", "dti"]

    if any(token in lower for token in anat_tokens) and not any(token in lower for token in func_tokens):
        hints["bids_dir"] = "anat"
        hints["bids_name"] = "T2w" if "t2" in lower else "T1w"

    elif any(token in lower for token in dwi_tokens):
        hints["bids_dir"] = "dwi"
        hints["bids_name"] = "dwi"

    elif any(token in lower for token in fmap_tokens):
        hints["bids_dir"] = "fmap"
        if "phasediff" in lower or "phase" in lower:
            hints["bids_name"] = "phasediff"
        elif "magnitude" in lower:
            hints["bids_name"] = "magnitude1"
        else:
            hints["bids_name"] = "phasediff"
        hints["intended_for"] = suggest_intended_for(sequence_name, all_sequence_names)

    elif any(token in lower for token in func_tokens):
        hints["bids_dir"] = "func"
        hints["bids_name"] = f"task-{infer_task_label(sequence_name)}_bold"

    return hints


def infer_task_label(sequence_name: str) -> str:
    lower = sequence_name.lower()
    if "rest" in lower:
        return "rest"
    if "nback" in lower:
        return "nback"
    if "flanker" in lower:
        return "flanker"

    parts = [part for part in lower.replace("-", " ").replace("_", " ").split() if part]
    common = [p for p in parts if p.isalpha() and p not in {"bold", "fmri", "task", "run"}]
    return common[0] if common else "unknown"


def suggest_intended_for(sequence_name: str, all_sequence_names: List[str]) -> str:
    func_candidates = [name for name in all_sequence_names if any(tok in name.lower() for tok in ["bold", "fmri", "rest", "nback", "flanker"])]
    if not func_candidates:
        return ""

    source_tokens = set(tokenize_name(sequence_name))
    best_match = ""
    best_score = 0
    for candidate in func_candidates:
        score = len(source_tokens.intersection(set(tokenize_name(candidate))))
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match or func_candidates[0]


def tokenize_name(name: str) -> List[str]:
    return [token for token in name.lower().replace("-", " ").replace("_", " ").split() if token]


def run_annotation_ui(protocol_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sequence_names = list(protocol_data.keys())
    entries = []
    for sequence_name, values in protocol_data.items():
        bids_dir, bids_name, intended_for = (values + ["", "", ""])[:3]
        if bids_dir in {"EXCLUDE_BIDS_Directory", "UNASSIGNED"}:
            bids_dir = "unknown"
        if bids_name in {"EXCLUDE_BIDS_Name", "UNASSIGNED"}:
            bids_name = ""
        entries.append(
            {
                "sequence_name": sequence_name,
                "bids_dir": bids_dir,
                "bids_name": bids_name,
                "intended_for": intended_for,
            }
        )

    for entry in entries:
        if entry["bids_dir"] == "unknown" or not entry["bids_name"]:
            entry.update(suggest_labels(entry["sequence_name"], sequence_names))

    if PROMPT_TOOLKIT_AVAILABLE:
        return _run_prompt_toolkit_ui(entries)
    return _run_text_ui(entries)


def _run_prompt_toolkit_ui(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    total = len(entries)
    current_index = 0

    while True:
        entry = entries[current_index]
        title = f"Sequence {current_index + 1} of {total}: {entry['sequence_name']}"
        message = (
            f"bids_dir: {entry['bids_dir']}\n"
            f"bids_name: {entry['bids_name']}\n"
            f"intended_for: {entry['intended_for'] or '<none>'}\n\n"
            "Select an action to update this sequence or move to another one."
        )
        buttons = [
            ("Edit bids_dir", "edit_dir"),
            ("Edit bids_name", "edit_name"),
            ("Edit intended_for", "edit_intended"),
            ("Previous sequence", "previous"),
            ("Next sequence", "next"),
            ("Save and exit", "save"),
            ("Quit without saving", "quit"),
        ]

        selection = button_dialog(
            title=title,
            text=message,
            buttons=buttons,
            style=DEFAULT_STYLE,
        ).run()

        if selection == "edit_dir":
            entry["bids_dir"] = _choose_bids_dir(entry["bids_dir"])
            if entry["bids_dir"] != "fmap":
                entry["intended_for"] = ""

        elif selection == "edit_name":
            entry["bids_name"] = _prompt_text("BIDS name", entry["bids_name"])

        elif selection == "edit_intended":
            entry["intended_for"] = _prompt_text("intended_for", entry["intended_for"])

        elif selection == "next":
            current_index = min(total - 1, current_index + 1)

        elif selection == "previous":
            current_index = max(0, current_index - 1)

        elif selection == "save":
            return {
                item["sequence_name"]: [item["bids_dir"], item["bids_name"], item["intended_for"] or ""]
                for item in entries
            }

        elif selection == "quit":
            raise KeyboardInterrupt("User exited without saving.")


def _run_text_ui(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    print("prompt_toolkit is not installed; falling back to plain text prompts.")
    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        print(f"\nSequence {index} of {total}: {entry['sequence_name']}")
        entry["bids_dir"] = _prompt_text("bids_dir", entry["bids_dir"])
        entry["bids_name"] = _prompt_text("bids_name", entry["bids_name"])
        if entry["bids_dir"] == "fmap":
            entry["intended_for"] = _prompt_text("intended_for", entry["intended_for"])
        else:
            entry["intended_for"] = ""
    return {
        entry["sequence_name"]: [entry["bids_dir"], entry["bids_name"], entry["intended_for"] or ""]
        for entry in entries
    }


def _choose_bids_dir(current_value: str) -> str:
    values = [(option, option) for option in BIDS_DIR_OPTIONS]
    result = radiolist_dialog(
        title="Select BIDS directory",
        text="Choose the most appropriate BIDS directory for this sequence:",
        values=values,
        style=DEFAULT_STYLE,
    ).run()
    return result or current_value


def _prompt_text(label: str, default: str) -> str:
    if PROMPT_TOOLKIT_AVAILABLE:
        response = input_dialog(title=f"Edit {label}", text=f"Enter value for {label}:", default=default).run()
        return response or default
    print(f"{label} [{default}]: ", end="")
    answer = input().strip()
    return answer if answer else default


def save_protocol(data: Dict[str, List[str]], file_path: str) -> None:
    path = Path(file_path)
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
    print(f"Saved updated protocol to {path}")
    print(f"Backup of original saved to {backup_path}")


def annotate_protocol(file_path: str) -> None:
    protocol_data = load_protocol(file_path)
    summarize_sequences(protocol_data)
    try:
        updated_data = run_annotation_ui(protocol_data)
    except KeyboardInterrupt:
        print("\nAnnotation cancelled. No changes were written.")
        return
    save_protocol(updated_data, file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.annotate_protocol /path/to/Protocol_Translator.json")
        sys.exit(1)
    annotate_protocol(sys.argv[1])
