from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import TaskConfig
from .models import EventTable
from .parsers.registry import ParserRegistry
from .parsers.stroop import StroopParser
from .reader import EPrimeReader
from .validator import validate_event_table
from .writer import EventsWriter


def run_behavior_events(
    subject: str,
    config: dict[str, Any],
    task_name: str | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    behavior_config = config.get("behavior", {})
    task_name = task_name or behavior_config.get("task") or "stroop"

    search_root = Path(behavior_config.get("edat3_search_root") or config.get("study_root", "."))
    if not search_root.is_absolute():
        search_root = Path(config.get("study_root", ".")) / search_root

    edat3_candidates = sorted(search_root.rglob("*.edat3")) if search_root.exists() else []
    if not edat3_candidates:
        raise FileNotFoundError(f"No .edat3 files found under {search_root}.")

    subject_label = str(subject).strip()
    if subject_label:
        edat3_candidates = [candidate for candidate in edat3_candidates if subject_label in candidate.as_posix()]
    if not edat3_candidates:
        edat3_candidates = sorted(search_root.rglob("*.edat3"))

    edat3_path = edat3_candidates[0]

    config_dir = Path(behavior_config.get("task_configs_dir") or config.get("code_dir", "code"))
    if not config_dir.is_absolute():
        config_dir = Path(config.get("study_root", ".")) / config_dir
    task_config_path = config_dir / f"{task_name}.json"
    if not task_config_path.exists():
        task_config_path = config_dir / f"task-{task_name}.json"
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config not found for '{task_name}' at {config_dir}.")

    task_config = TaskConfig.load(task_config_path)
    reader = EPrimeReader(backend=behavior_config.get("reader_backend", "auto"))
    raw_df = reader.read(edat3_path)

    registry = ParserRegistry()
    registry.register("stroop", StroopParser)
    plugin_dir = behavior_config.get("parser_plugins_dir")
    if plugin_dir:
        plugin_path = Path(plugin_dir)
        if not plugin_path.is_absolute():
            plugin_path = Path(config.get("study_root", ".")) / plugin_path
        registry.load_plugins(plugin_path)
    parser_cls = registry.get(task_name)
    parser = parser_cls()
    events_df = parser.parse(raw_df, task_config)

    validated = validate_event_table(events_df)
    event_table = EventTable.from_records(validated.to_dict(orient="records"))

    output_root = Path(behavior_config.get("output_dir") or "derivatives/behavior/events")
    if not output_root.is_absolute():
        output_root = Path(config.get("study_root", ".")) / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    output_path = output_root / subject_label / f"task-{task_name}_events.tsv"
    sidecar_path = output_root / subject_label / f"task-{task_name}_events.json"
    if dry_run:
        return {
            "success": True,
            "skipped": False,
            "message": f"Dry run: would write behavior events for {subject_label} to {output_path}",
            "command": "",
            "returncode": None,
        }

    writer = EventsWriter()
    writer.write(validated, output_path, sidecar_path=sidecar_path, task_name=task_name, metadata=task_config.metadata)

    return {
        "success": True,
        "skipped": False,
        "message": f"Behavior events written to {output_path}",
        "command": "",
        "returncode": 0,
    }
