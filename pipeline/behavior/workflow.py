from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from .config import TaskConfig
from .models import EventTable
from .parsers.registry import ParserRegistry
from .parsers.stroop import StroopParser
from .reader import EPrimeReader
from .validator import validate_event_table
from .writer import EventsWriter


def _log_verbose(behavior_config: dict[str, Any], message: str, *args: Any) -> None:
    if not behavior_config.get("verbose", False):
        return
    print(f"[behavior][verbose] {message % args if args else message}")


def run_behavior_events(
    subject: str,
    config: dict[str, Any],
    task_name: str | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    behavior_config = config.get("behavior", {})
    task_name = task_name or behavior_config.get("task") or "stroop"
    _log_verbose(behavior_config, "Starting behavior events workflow for subject '%s'", subject)
    _log_verbose(behavior_config, "Requested task: %s", task_name)

    try:
        search_root = Path(behavior_config.get("edat3_search_root") or config.get("study_root", "."))
        if not search_root.is_absolute():
            search_root = Path(config.get("study_root", ".")) / search_root

        _log_verbose(behavior_config, "Searching for .edat3 files under: %s", search_root)
        edat3_candidates = sorted(search_root.rglob("*.edat3")) if search_root.exists() else []
        _log_verbose(behavior_config, "Initial .edat3 file candidates: %s", edat3_candidates)
        if not edat3_candidates:
            raise FileNotFoundError(f"No .edat3 files found under {search_root}.")

        subject_label = str(subject).strip()
        if subject_label:
            _log_verbose(behavior_config, "Filtering .edat3 candidates for subject '%s'", subject_label)
            edat3_candidates = [candidate for candidate in edat3_candidates if subject_label in candidate.as_posix()]
        _log_verbose(behavior_config, "Filtered .edat3 candidates: %s", edat3_candidates)
        if not edat3_candidates:
            _log_verbose(behavior_config, "No .edat3 files matched subject '%s'; falling back to all candidates", subject_label)
            edat3_candidates = sorted(search_root.rglob("*.edat3"))

        edat3_path = edat3_candidates[0]
        _log_verbose(behavior_config, "Using .edat3 file: %s", edat3_path)

        config_dir = Path(behavior_config.get("task_configs_dir") or config.get("code_dir", "code"))
        if not config_dir.is_absolute():
            config_dir = Path(config.get("study_root", ".")) / config_dir
        _log_verbose(behavior_config, "Searching for task config in: %s", config_dir)
        task_config_path = config_dir / f"{task_name}.json"
        if not task_config_path.exists():
            task_config_path = config_dir / f"task-{task_name}.json"
        if not task_config_path.exists():
            raise FileNotFoundError(f"Task config not found for '{task_name}' at {config_dir}.")

        _log_verbose(behavior_config, "Loading task config from: %s", task_config_path)
        task_config = TaskConfig.load(task_config_path)
        _log_verbose(behavior_config, "Task config columns: %s", task_config.column_sources())
        _log_verbose(behavior_config, "Task config optional columns: %s", task_config.optional_columns)
        _log_verbose(behavior_config, "Task config metadata: %s", task_config.metadata)
        reader = EPrimeReader(backend=behavior_config.get("reader_backend", "auto"))
        _log_verbose(behavior_config, "Reader backend: %s", behavior_config.get("reader_backend", "auto"))
        raw_df = reader.read(edat3_path)
        _log_verbose(behavior_config, "Loaded raw E-Prime dataframe with shape: %s", raw_df.shape)

        registry = ParserRegistry()
        registry.register("stroop", StroopParser)
        plugin_dir = behavior_config.get("parser_plugins_dir")
        if plugin_dir:
            plugin_path = Path(plugin_dir)
            if not plugin_path.is_absolute():
                plugin_path = Path(config.get("study_root", ".")) / plugin_path
            _log_verbose(behavior_config, "Loading parser plugins from: %s", plugin_path)
            registry.load_plugins(plugin_path)
        _log_verbose(behavior_config, "Using parser for task '%s'", task_name)
        parser_cls = registry.get(task_name)
        parser = parser_cls()
        events_df = parser.parse(raw_df, task_config)
        _log_verbose(behavior_config, "Parsed events dataframe with shape: %s", events_df.shape)
        _log_verbose(behavior_config, "Parsed events preview: %s", events_df.head(5).to_dict(orient="records"))

        validated = validate_event_table(events_df)
        _log_verbose(behavior_config, "Validated events dataframe with shape: %s", validated.shape)
        EventTable.from_records(validated.to_dict(orient="records"))

        output_root = Path(behavior_config.get("output_dir") or "derivatives/behavior/events")
        if not output_root.is_absolute():
            output_root = Path(config.get("study_root", ".")) / output_root
        output_root.mkdir(parents=True, exist_ok=True)
        _log_verbose(behavior_config, "Writing events outputs under: %s", output_root)

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
        _log_verbose(behavior_config, "Wrote events TSV to: %s", output_path)
        _log_verbose(behavior_config, "Wrote events JSON sidecar to: %s", sidecar_path)

        return {
            "success": True,
            "skipped": False,
            "message": f"Behavior events written to {output_path}",
            "command": "",
            "returncode": 0,
        }
    except Exception as exc:
        if behavior_config.get("verbose", False):
            print(f"[behavior][verbose] Events workflow failed: {exc}")
            traceback.print_exc()
        raise
