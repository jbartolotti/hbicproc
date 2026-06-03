import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import questionary  # type: ignore
    QUESTIONARY_AVAILABLE = True
except ImportError:
    QUESTIONARY_AVAILABLE = False

from .config import load_config, load_default_config
from .logger import append_event

BIDS_DIR_OPTIONS = ["EXCLUDE_BIDS_Directory", "anat", "func", "fmap", "dwi"]
EXCLUDE_BIDS_NAME = "EXCLUDE_BIDS_Name"
UNASSIGNED = "UNASSIGNED"


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


def suggest_labels(sequence_name: str, all_sequence_names: List[str], tokens: Dict[str, List[str]]) -> Dict[str, str]:
    lower = sequence_name.lower()
    hints = {
        "bids_dir": "EXCLUDE_BIDS_Directory",
        "bids_name": EXCLUDE_BIDS_NAME,
        "intended_for": UNASSIGNED,
    }

    if "aahead" in lower or "aahead" in lower or "scout" in lower or "localizer" in lower:
        return hints

    anat_tokens = tokens.get("anat", [])
    func_tokens = tokens.get("func", [])
    fmap_tokens = tokens.get("fmap", [])
    dwi_tokens = tokens.get("dwi", [])

    if any(token in lower for token in anat_tokens) and not any(token in lower for token in func_tokens):
        hints["bids_dir"] = "anat"
        hints["bids_name"] = "T1w" if "mprage" in lower or "t1" in lower else "T1w"

    elif any(token in lower for token in dwi_tokens):
        hints["bids_dir"] = "dwi"
        hints["bids_name"] = "dwi"

    elif any(token in lower for token in fmap_tokens):
        hints["bids_dir"] = "fmap"
        hints["bids_name"] = infer_fieldmap_name(lower)
        hints["intended_for"] = suggest_intended_for(sequence_name, all_sequence_names, tokens)

    elif any(token in lower for token in func_tokens):
        hints["bids_dir"] = "func"
        hints["bids_name"] = infer_func_name(sequence_name, tokens)

    return hints


def infer_task_label(sequence_name: str, tokens: Dict[str, List[str]]) -> str:
    lower = sequence_name.lower()
    func_tokens = tokens.get("func", [])
    task_tokens = [tok for tok in func_tokens if tok not in {"bold", "fmri", "task", "sbref", "run"}]

    for tok in task_tokens:
        if tok in lower:
            return tok

    parts = [part for part in lower.replace("-", " ").replace("_", " ").split() if part]
    ignore = set(func_tokens) | {"bold", "fmri", "task", "run"}
    common = [p for p in parts if p.isalpha() and p not in ignore]
    return common[0] if common else "unknown"


def infer_func_name(sequence_name: str, tokens: Dict[str, List[str]]) -> str:
    lower = sequence_name.lower()
    if "sbref" in lower:
        return f"task-{infer_task_label(sequence_name, tokens)}"
    return f"task-{infer_task_label(sequence_name, tokens)}_bold"


def infer_fieldmap_name(lower_sequence_name: str) -> str:
    tokens = tokenize_name(lower_sequence_name)
    if "ap" in tokens:
        return "dir-AP_epi"
    if "pa" in tokens:
        return "dir-PA_epi"
    if "phasediff" in lower_sequence_name or "phase" in lower_sequence_name:
        return "dir-AP_epi"
    return "dir-PA_epi"


def suggest_intended_for(sequence_name: str, all_sequence_names: List[str], tokens: Dict[str, List[str]]) -> str:
    func_candidates = [name for name in all_sequence_names if any(tok in name.lower() for tok in tokens.get("func", []))]
    if not func_candidates:
        return UNASSIGNED

    candidate_labels = []
    for name in func_candidates:
        func_name = infer_func_name(name, tokens)
        if func_name not in {"task-unknown_bold", "task-unknown_sbref"}:
            candidate_labels.append(func_name)

    if not candidate_labels:
        return UNASSIGNED

    task_tokens = [tok for tok in tokens.get("func", []) if tok not in {"bold", "fmri", "task", "sbref", "run"}]
    lower = sequence_name.lower()
    for tok in task_tokens:
        if tok in lower:
            candidate = f"task-{tok}_sbref" if "sbref" in lower else f"task-{tok}_bold"
            if candidate in candidate_labels:
                return candidate

    source_tokens = set(tokenize_name(sequence_name))
    best_label = None
    best_score = -1
    for label in candidate_labels:
        label_tokens = set(tokenize_name(label.replace("task-", "")))
        score = len(source_tokens & label_tokens)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label or candidate_labels[0]


def tokenize_name(name: str) -> List[str]:
    return [token for token in name.lower().replace("-", " ").replace("_", " ").split() if token]


def run_annotation_ui(protocol_data: Dict[str, List[str]], tokens: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sequence_names = list(protocol_data.keys())
    entries = []
    for sequence_name, values in protocol_data.items():
        bids_dir, bids_name, intended_for = (values + [EXCLUDE_BIDS_NAME, UNASSIGNED])[:3]
        if isinstance(intended_for, list):
            intended_for = intended_for[0] if intended_for else UNASSIGNED
        if bids_dir not in BIDS_DIR_OPTIONS:
            bids_dir = "EXCLUDE_BIDS_Directory"
        if bids_name == "" or bids_name == "UNASSIGNED":
            bids_name = EXCLUDE_BIDS_NAME
        if intended_for == "" or intended_for is None:
            intended_for = UNASSIGNED

        entry = {
            "sequence_name": sequence_name,
            "bids_dir": bids_dir,
            "bids_name": bids_name,
            "intended_for": intended_for,
        }
        if bids_dir == "EXCLUDE_BIDS_Directory" and bids_name == EXCLUDE_BIDS_NAME and intended_for == UNASSIGNED:
            entry.update(suggest_labels(sequence_name, sequence_names, tokens))
        entries.append(entry)

    if QUESTIONARY_AVAILABLE:
        return _run_questionary_ui(entries, tokens)
    return _run_text_ui(entries)


def _run_questionary_ui(entries: List[Dict[str, Any]], tokens: Dict[str, List[str]]) -> Dict[str, List[str]]:
    total = len(entries)
    current_index = 0

    while True:
        entry = entries[current_index]
        print(f"\nSequence {current_index + 1} of {total}: {entry['sequence_name']}")
        print(f"  bids_dir: {entry['bids_dir']}")
        print(f"  bids_name: {entry['bids_name']}")
        print(f"  intended_for: {entry['intended_for']}\n")

        action = questionary.select(
            "Choose an action:",
            choices=[
                "Edit bids_dir",
                "Edit bids_name",
                "Edit intended_for",
                "Previous sequence",
                "Next sequence",
                "Save and exit",
                "Quit without saving",
            ],
            default="Next sequence",
        ).ask()
        if action is None:
            raise KeyboardInterrupt("User exited without saving.")

        if action == "Edit bids_dir":
            entry["bids_dir"] = questionary.select(
                "Select bids_dir:",
                choices=BIDS_DIR_OPTIONS,
                default=entry["bids_dir"],
            ).ask() or entry["bids_dir"]
            if entry["bids_dir"] == "EXCLUDE_BIDS_Directory":
                entry["bids_name"] = EXCLUDE_BIDS_NAME
                entry["intended_for"] = UNASSIGNED
            elif entry["bids_dir"] != "fmap":
                entry["intended_for"] = UNASSIGNED
            if entry["bids_name"] == EXCLUDE_BIDS_NAME:
                entry["bids_name"] = suggest_labels(entry["sequence_name"], [e["sequence_name"] for e in entries], tokens)["bids_name"]
        elif action == "Edit bids_name":
            choices = _bids_name_options(entry, tokens)
            if "Custom..." not in choices:
                choices.append("Custom...")
            choice = questionary.select(
                "Select bids_name:",
                choices=choices,
                default=entry["bids_name"],
            ).ask()
            if choice == "Custom...":
                custom_value = questionary.text(
                    "Enter custom bids_name:",
                    default=entry["bids_name"] if entry["bids_name"] not in [EXCLUDE_BIDS_NAME, UNASSIGNED] else "",
                ).ask()
                entry["bids_name"] = custom_value or entry["bids_name"]
            else:
                entry["bids_name"] = choice or entry["bids_name"]
            if entry["bids_name"] == EXCLUDE_BIDS_NAME and entry["bids_dir"] != "EXCLUDE_BIDS_Directory":
                entry["bids_dir"] = "EXCLUDE_BIDS_Directory"

        elif action == "Edit intended_for":
            choices = _intended_for_options(entry, entries)
            if "Custom..." not in choices:
                choices.append("Custom...")
            choice = questionary.select(
                "Select intended_for:",
                choices=choices,
                default=entry["intended_for"],
            ).ask()
            if choice == "Custom...":
                custom_value = questionary.text(
                    "Enter custom intended_for:",
                    default=entry["intended_for"] if entry["intended_for"] != UNASSIGNED else "",
                ).ask()
                entry["intended_for"] = custom_value or entry["intended_for"]
            else:
                entry["intended_for"] = choice or entry["intended_for"]
            if entry["bids_dir"] != "fmap":
                entry["intended_for"] = UNASSIGNED

        elif action == "Next sequence":
            current_index = min(total - 1, current_index + 1)

        elif action == "Previous sequence":
            current_index = max(0, current_index - 1)

        elif action == "Save and exit":
            return {item["sequence_name"]: [item["bids_dir"], item["bids_name"], item["intended_for"]] for item in entries}

        elif action == "Quit without saving":
            raise KeyboardInterrupt("User exited without saving.")


def _run_text_ui(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    print("questionary is not installed; falling back to plain text prompts.")
    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        print(f"\nSequence {index} of {total}: {entry['sequence_name']}")
        entry["bids_dir"] = _prompt_text("bids_dir", entry["bids_dir"])
        entry["bids_name"] = _prompt_text("bids_name", entry["bids_name"])
        if entry["bids_dir"] == "fmap":
            entry["intended_for"] = _prompt_text("intended_for", entry["intended_for"])
        else:
            entry["intended_for"] = UNASSIGNED
    return {
        entry["sequence_name"]: [entry["bids_dir"], entry["bids_name"], entry["intended_for"]]
        for entry in entries
    }


def _bids_name_options(entry: Dict[str, Any], tokens: Dict[str, List[str]]) -> List[str]:
    if entry["bids_dir"] == "anat":
        choices = ["T1w", "T2w", EXCLUDE_BIDS_NAME]
    elif entry["bids_dir"] == "func":
        base = infer_task_label(entry["sequence_name"], tokens)
        if "sbref" in entry["sequence_name"].lower():
            choices = [f"task-{base}_sbref", EXCLUDE_BIDS_NAME]
        else:
            choices = [f"task-{base}_bold", EXCLUDE_BIDS_NAME]
        func_tokens = [tok for tok in tokens.get("func", []) if tok not in {"bold", "fmri", "task", "sbref", "run"}]
        for tok in func_tokens:
            if tok != base:
                if "sbref" in entry["sequence_name"].lower():
                    choices.insert(-1, f"task-{tok}_sbref")
                else:
                    choices.insert(-1, f"task-{tok}_bold")
    elif entry["bids_dir"] == "fmap":
        choices = ["dir-AP_epi", "dir-PA_epi", EXCLUDE_BIDS_NAME]
    elif entry["bids_dir"] == "dwi":
        choices = ["dwi", EXCLUDE_BIDS_NAME]
    else:
        choices = [EXCLUDE_BIDS_NAME]
    if entry["bids_name"] not in choices:
        choices.insert(0, entry["bids_name"])
    return choices


def _intended_for_options(entry: Dict[str, Any], entries: List[Dict[str, Any]]) -> List[str]:
    if entry["bids_dir"] != "fmap":
        return [UNASSIGNED]

    func_labels = sorted({item["bids_name"] for item in entries if item["bids_dir"] == "func" and item["bids_name"] != EXCLUDE_BIDS_NAME})
    if not func_labels:
        return [UNASSIGNED]
    choices = func_labels + [UNASSIGNED]
    if entry["intended_for"] not in choices:
        choices.insert(0, entry["intended_for"])
    return choices


def _prompt_text(label: str, default: str) -> str:
    print(f"{label} [{default}]: ", end="")
    answer = input().strip()
    return answer if answer else default


def save_protocol(data: Dict[str, List[str]], file_path: str) -> None:
    path = Path(file_path)
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    normalized = {}
    for sequence_name, values in data.items():
        bids_dir, bids_name, intended_for = (values + [EXCLUDE_BIDS_NAME, UNASSIGNED])[:3]
        if intended_for != UNASSIGNED and not isinstance(intended_for, list):
            intended_for = [intended_for]
        normalized[sequence_name] = [bids_dir, bids_name, intended_for]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=4)
    print(f"Saved updated protocol to {path}")
    print(f"Backup of original saved to {backup_path}")


def annotate_protocol(file_path: str, config_path: Optional[str] = "pipeline_config.json") -> None:
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        config = load_default_config(str(Path(file_path).parent))
    append_event(f"Starting annotation for {file_path}", config, step="annotate_protocol")
    protocol_data = load_protocol(file_path)
    summarize_sequences(protocol_data)
    try:
        updated_data = run_annotation_ui(protocol_data, config.get("tokens", {}))
    except KeyboardInterrupt:
        append_event("Annotation cancelled by user", config, step="annotate_protocol")
        print("\nAnnotation cancelled. No changes were written.")
        return
    save_protocol(updated_data, file_path)
    append_event(f"Saved annotation to {file_path}", config, step="annotate_protocol")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Annotate a bidskit protocol_translator.json file.")
    parser.add_argument("file_path", help="Path to Protocol_Translator.json")
    parser.add_argument(
        "--config",
        default="pipeline_config.json",
        help="Path to a pipeline config JSON file.",
    )
    args = parser.parse_args(argv)

    annotate_protocol(args.file_path, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
