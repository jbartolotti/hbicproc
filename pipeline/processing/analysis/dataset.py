from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from typing import Any

from bids import BIDSLayout, BIDSLayoutIndexer

from ...core.paths import get_bids_root
from .context import InputDataset, TaskRunContext
from ...state import invalidate_bids_index, load_pipeline_state, pipeline_state_file, save_pipeline_state

logger = logging.getLogger(__name__)


_ENTITY_ALIASES = {
    "subject": ("subject", "sub"),
    "session": ("session", "ses"),
    "task": ("task",),
    "run": ("run",),
    "desc": ("desc",),
    "stat": ("stat",),
}


def _normalize_entity_text(value: Any, *, prefix: str | None = None) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    prefix_variants = []
    if prefix:
        prefix_variants.extend([f"{prefix}-", f"{prefix}_", prefix])

    for candidate in prefix_variants:
        if text.lower().startswith(candidate.lower()):
            text = text[len(candidate) :]
            break

    normalized = text.replace("_", "-")
    if not normalized:
        return None
    return normalized


class DatasetIndex:
    """Central BIDS-aware dataset index for discovery-oriented analysis plugins."""

    def __init__(
        self,
        bids_root: str | Path | None = None,
        *,
        study_root: str | Path | None = None,
        input_dataset: InputDataset | None = None,
    ) -> None:
        root = Path(bids_root) if bids_root is not None else (Path(study_root) if study_root is not None else Path("."))
        self.bids_root = root.resolve() if root.exists() else root
        self.study_root = Path(study_root).resolve() if study_root is not None and Path(study_root).exists() else self.bids_root
        self.input_dataset = input_dataset or InputDataset(
            name="fmriprep",
            path=self.bids_root / "derivatives" / "fmriprep",
        )
        derivative_path = Path(self.input_dataset.path)
        self.derivative_root = derivative_path.resolve() if derivative_path.exists() else derivative_path

        self.layout: BIDSLayout | None = None
        self.derivative_layout: BIDSLayout | None = None
        self._derivative_layouts: dict[Path, BIDSLayout | None] = {}
        self._task_runs_cache: dict[
            tuple[str | None, str | None, str | None, str | None, str],
            tuple[TaskRunContext, ...],
        ] = {}
        logger.info(
            "BIDS indexing: starting shared raw and derivative BIDSLayout construction (raw_root=%s derivative_root=%s)",
            self.bids_root,
            self.derivative_root,
        )
        if self.bids_root.exists():
            try:
                self.layout = self._build_cached_layout(
                    self.bids_root,
                    index_name="raw",
                    derivatives=False,
                    is_derivative=False,
                )
            except Exception:
                self.layout = None
        if self.derivative_root.exists():
            self.derivative_layout = self._build_derivative_layout(self.derivative_root)
        self._derivative_layouts[self.derivative_root] = self.derivative_layout
        logger.info(
            "BIDS indexing: shared layouts ready (raw_layout=%s derivative_layout=%s)",
            self.layout is not None,
            self.derivative_layout is not None,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        task_config: dict[str, Any] | None = None,
    ) -> "DatasetIndex":
        bids_root = config.get("bids_root") or config.get("study_root") or "."
        study_root = config.get("study_root") or bids_root
        analysis = config.get("analysis", {})
        input_dataset = analysis.get("input_dataset") if isinstance(analysis, dict) else None
        if task_config and "input_dataset" in task_config:
            input_dataset = task_config["input_dataset"]
        if not isinstance(input_dataset, dict):
            raise ValueError("analysis.input_dataset must be an object with 'name' and 'path'.")
        dataset_name = str(input_dataset.get("name", "")).strip()
        dataset_path = str(input_dataset.get("path", "")).strip()
        if not dataset_name or not dataset_path:
            raise ValueError("analysis.input_dataset must define non-empty 'name' and 'path'.")
        return cls(
            bids_root=bids_root,
            study_root=study_root,
            input_dataset=InputDataset(dataset_name, Path(dataset_path)),
        )

    def get_task_runs(
        self,
        *,
        subject: str | None = None,
        task: str | None = None,
        session: str | None = None,
        run: str | None = None,
        input_dataset: InputDataset | None = None,
    ) -> list[TaskRunContext]:
        normalized_subject = _normalize_entity_text(subject, prefix="sub") if subject is not None else None
        normalized_task = _normalize_entity_text(task, prefix="task") if task is not None else None
        normalized_session = _normalize_entity_text(session, prefix="ses") if session is not None else None
        normalized_run = _normalize_entity_text(run, prefix="run") if run is not None else None
        selected_dataset = input_dataset or self.input_dataset
        selected_root = Path(selected_dataset.path)
        cache_key = (
            normalized_subject,
            normalized_task,
            normalized_session,
            normalized_run,
            str(selected_root),
        )
        cached_runs = self._task_runs_cache.get(cache_key)
        if cached_runs is not None:
            logger.info(
                "Reusing cached dataset discovery: subject=%s task=%s session=%s run=%s runs=%d",
                normalized_subject,
                normalized_task,
                normalized_session,
                normalized_run,
                len(cached_runs),
            )
            return list(cached_runs)

        derivative_root = selected_root
        derivative_layout = self._get_derivative_layout(derivative_root, selected_dataset.name)

        if self.layout is None or derivative_layout is None:
            logger.info(
                "Dataset discovery skipped for subject=%s task=%s: raw_layout=%s derivative_layout=%s derivative_dataset=%s derivative_root=%s",
                subject,
                task,
                self.layout is not None,
                derivative_layout is not None,
                selected_dataset.name,
                derivative_root,
            )
            self._task_runs_cache[cache_key] = ()
            return []

        logger.info(
            "Dataset discovery requested: subject=%s task=%s session=%s run=%s (normalized: subject=%s task=%s session=%s run=%s)",
            subject,
            task,
            session,
            run,
            normalized_subject,
            normalized_task,
            normalized_session,
            normalized_run,
        )

        bold_records = self._query(
            derivative_layout,
            suffix="bold",
            extension=[".nii.gz", ".nii"],
            subject=normalized_subject,
            task=normalized_task,
            session=normalized_session,
            run=normalized_run,
        )

        events_records = self._query(
            self.layout,
            suffix="events",
            extension=".tsv",
            subject=normalized_subject,
            task=normalized_task,
            session=normalized_session,
            run=normalized_run,
        )

        confounds_records = self._query(
            derivative_layout,
            suffix="timeseries",
            extension=".tsv",
            subject=normalized_subject,
            task=normalized_task,
            session=normalized_session,
            run=normalized_run,
        )

        logger.info(
            "Dataset discovery counts: subject=%s task=%s BOLD=%d EVENTS=%d CONFOUNDS=%d",
            normalized_subject,
            normalized_task,
            len(bold_records),
            len(events_records),
            len(confounds_records),
        )
        logger.info("Discovered BOLD files: %s", [str(getattr(record, 'path', record)) for record in bold_records])
        logger.info("Discovered events files: %s", [str(getattr(record, 'path', record)) for record in events_records])
        logger.info("Discovered confounds files: %s", [str(getattr(record, 'path', record)) for record in confounds_records])

        runs: list[TaskRunContext] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()

        for bold_record in bold_records:
            bold_info = self._record_entities(bold_record)
            subject_value = self._entity_value(bold_info, "subject")
            task_value = self._entity_value(bold_info, "task")
            session_value = self._entity_value(bold_info, "session")
            run_value = self._entity_value(bold_info, "run")
            logger.info(
                "Candidate BOLD: subject=%s session=%s task=%s run=%s path=%s",
                subject_value,
                session_value,
                task_value,
                run_value,
                bold_record.path,
            )
            if not self._matches_expected(bold_info, {"subject": normalized_subject, "task": normalized_task, "session": normalized_session, "run": normalized_run}):
                logger.info(
                    "Rejecting candidate BOLD: %s",
                    self._mismatch_summary(
                        bold_info,
                        {"subject": normalized_subject, "task": normalized_task, "session": normalized_session, "run": normalized_run},
                    ),
                )
                continue

            event_match = self._first_match(
                events_records,
                {
                    "subject": subject_value,
                    "task": task_value,
                    "session": session_value,
                    "run": run_value,
                },
                label="EVENTS",
            )
            confounds_match = self._first_match(
                confounds_records,
                {
                    "subject": subject_value,
                    "task": task_value,
                    "session": session_value,
                    "run": run_value,
                    "desc": "confounds",
                },
                label="CONFOUNDS",
            )
            if event_match is None or confounds_match is None:
                logger.info(
                    "Rejecting candidate: BOLD subject=%s session=%s task=%s run=%s; EVENTS=%s CONFOUNDS=%s",
                    subject_value,
                    session_value,
                    task_value,
                    run_value,
                    self._candidate_summary(event_match),
                    self._candidate_summary(confounds_match),
                )
                continue

            key = (subject_value or "", task_value or "", session_value or None, run_value or None)
            if key in seen:
                continue
            seen.add(key)

            resolved_run = TaskRunContext(
                subject=subject_value or "",
                task=task_value or "",
                session=session_value,
                run=run_value,
                bold_path=Path(str(bold_record.path)),
                events_path=Path(str(event_match.path)),
                confounds_path=Path(str(confounds_match.path)),
                input_dataset=selected_dataset,
            )
            runs.append(resolved_run)

        logger.info("Resolved task run contexts for subject=%s task=%s: %s", normalized_subject, normalized_task, [run.as_dict() for run in runs])
        resolved_runs = tuple(sorted(runs, key=lambda item: (item.session or "", item.run or "")))
        self._task_runs_cache[cache_key] = resolved_runs
        return list(resolved_runs)

    def _get_derivative_layout(self, derivative_root: Path, index_name: str | None = None) -> BIDSLayout | None:
        if derivative_root not in self._derivative_layouts:
            self._derivative_layouts[derivative_root] = self._build_derivative_layout(
                derivative_root,
                index_name=index_name or self.input_dataset.name,
            )
        return self._derivative_layouts[derivative_root]

    def _build_derivative_layout(
        self,
        derivative_root: Path,
        *,
        index_name: str | None = None,
    ) -> BIDSLayout | None:
        if not derivative_root.exists():
            return None
        try:
            return self._build_cached_layout(
                derivative_root,
                index_name=index_name or self.input_dataset.name,
                derivatives=False,
                is_derivative=True,
            )
        except Exception:
            return None

    def _build_cached_layout(
        self,
        root: Path,
        *,
        index_name: str,
        derivatives: bool,
        is_derivative: bool,
    ) -> BIDSLayout:
        cache_dir = self.bids_root / "code" / "cache" / "pybids"
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = json.dumps(
            {"name": index_name, "path": str(root.resolve())},
            sort_keys=True,
        ).encode("utf-8")
        identity_hash = hashlib.sha256(identity).hexdigest()[:16]
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in index_name)
        database_path = cache_dir / f"{safe_name}-{identity_hash}"
        metadata_path = cache_dir / f"{safe_name}-{identity_hash}.json"
        state_path = pipeline_state_file(self.bids_root)
        state = load_pipeline_state(self.bids_root)
        current_revision = state.get("bids_indexes", {}).get(index_name, 0)
        try:
            current_revision = int(current_revision)
        except (TypeError, ValueError):
            current_revision = 0

        cached_revision = None
        if state_path.exists() and database_path.exists() and metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                cached_revision = int(metadata.get("revision"))
            except (OSError, ValueError, TypeError):
                cached_revision = None

        layout_kwargs = {
            "validate": False,
            "derivatives": derivatives,
            "is_derivative": is_derivative,
            "indexer": BIDSLayoutIndexer(validate=False),
            "database_path": str(database_path),
        }
        if cached_revision == current_revision:
            logger.info(
                "Using cached PyBIDS index %s (revision=%d, database=%s)",
                index_name,
                current_revision,
                database_path,
            )
            return BIDSLayout(str(root), **layout_kwargs)

        if database_path.exists():
            logger.info(
                "PyBIDS cache stale, rebuilding %s (cached_revision=%s current_revision=%d)",
                index_name,
                cached_revision,
                current_revision,
            )
        else:
            logger.info("Created new PyBIDS cache %s (revision=%d)", index_name, current_revision)
        layout_kwargs["reset_database"] = True
        layout = BIDSLayout(str(root), **layout_kwargs)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"index_name": index_name, "revision": current_revision, "identity": identity.decode("utf-8")},
                handle,
                indent=2,
            )
            handle.write("\n")
        state.setdefault("bids_indexes", {})[index_name] = current_revision
        save_pipeline_state(self.bids_root, state)
        return layout

    def _query(self, layout: BIDSLayout | None, **filters: Any) -> list[Any]:
        if layout is None:
            return []
        query: dict[str, Any] = {"return_type": "object"}
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)) and not value:
                continue
            query[key] = value
        try:
            return list(layout.get(**query))
        except TypeError:
            query.pop("return_type", None)
            return list(layout.get(**query))

    def _candidate_summary(self, record: Any) -> str:
        if record is None:
            return "none"
        entities = self._record_entities(record)
        return (
            f"subject={self._entity_value(entities, 'subject')} "
            f"session={self._entity_value(entities, 'session')} "
            f"task={self._entity_value(entities, 'task')} "
            f"run={self._entity_value(entities, 'run')} "
            f"path={getattr(record, 'path', record)}"
        )

    def _record_entities(self, record: Any) -> dict[str, Any]:
        if record is None:
            return {}
        if hasattr(record, "get_entities"):
            entities = record.get_entities()
        elif hasattr(record, "entities"):
            entities = record.entities
        elif isinstance(record, dict):
            entities = record
        else:
            entities = {}
        if not isinstance(entities, dict):
            entities = {}

        normalized = {str(key).lower(): value for key, value in entities.items()}
        path_value = self._path_entities(record)
        if path_value:
            normalized.update(path_value)
        return normalized

    def _path_entities(self, record: Any) -> dict[str, str]:
        path = getattr(record, "path", None)
        if path is None and isinstance(record, dict):
            path = record.get("path") or record.get("filename")
        if not path:
            return {}

        name = Path(str(path)).name
        for suffix in (".nii.gz", ".nii", ".tsv", ".json", ".csv"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        entities: dict[str, str] = {}
        for token in name.split("_"):
            if not token or "-" not in token:
                continue
            key, value = token.split("-", 1)
            key = key.strip().lower()
            normalized_key = {
                "sub": "subject",
                "ses": "session",
                "task": "task",
                "run": "run",
                "desc": "desc",
                "stat": "stat",
            }.get(key, key)
            entities[normalized_key] = _normalize_entity_text(value, prefix=key)
        return {key: value for key, value in entities.items() if value is not None}

    def _entity_value(self, entities: dict[str, Any], name: str) -> str | None:
        aliases = _ENTITY_ALIASES.get(name, (name,))
        for alias in aliases:
            for key in (alias, alias.lower()):
                if key in entities:
                    value = entities.get(key)
                    normalized = _normalize_entity_text(
                        value,
                        prefix={"subject": "sub", "session": "ses", "task": "task", "run": "run", "desc": "desc", "stat": "stat"}.get(name, name),
                    )
                    if normalized is not None:
                        return normalized
        return None

    def _matches_expected(
        self,
        entities: dict[str, Any],
        expected: dict[str, str | None],
    ) -> bool:
        for key, expected_value in expected.items():
            if expected_value is None:
                continue
            actual_value = self._entity_value(entities, key)
            if actual_value != expected_value:
                return False
        return True

    def _first_match(
        self,
        records: list[Any],
        expected: dict[str, str | None],
        *,
        label: str,
    ) -> Any | None:
        for record in records:
            entities = self._record_entities(record)
            actual = {
                key: self._entity_value(entities, key)
                for key in expected
            }
            logger.info(
                "Candidate %s: subject=%s session=%s task=%s run=%s path=%s",
                label,
                actual.get("subject"),
                actual.get("session"),
                actual.get("task"),
                actual.get("run"),
                getattr(record, "path", record),
            )

        for record in records:
            entities = self._record_entities(record)
            if self._matches_expected(entities, expected):
                return record
            logger.info("Rejecting candidate %s: %s", label, self._mismatch_summary(entities, expected))
        return None

    def _mismatch_summary(
        self,
        entities: dict[str, Any],
        expected: dict[str, str | None],
    ) -> str:
        mismatches = []
        for key, expected_value in expected.items():
            if expected_value is None:
                continue
            actual_value = self._entity_value(entities, key)
            if actual_value != expected_value:
                mismatches.append(f"{key}={actual_value} expected={expected_value}")
        return "; ".join(mismatches) or "no entity mismatch"


def get_dataset_index(config: dict[str, Any]) -> DatasetIndex:
    return DatasetIndex.from_config(config)


def regenerate_cached_indexes(config: dict[str, Any]) -> None:
    """Rebuild the raw and configured derivative PyBIDS indexes."""

    bids_root = get_bids_root(config)
    datasets: dict[tuple[str, str], dict[str, str]] = {}
    cache_dir = bids_root / "code" / "cache" / "pybids"
    for metadata_path in cache_dir.glob("*.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            identity = json.loads(metadata.get("identity", "{}"))
            name = str(metadata.get("index_name", "")).strip()
            path = str(identity.get("path", "")).strip()
            if name and name != "raw" and path:
                datasets[(name, path)] = {"name": name, "path": path}
        except (OSError, TypeError, ValueError):
            continue

    analysis = config.get("analysis", {})
    if isinstance(analysis, dict):
        global_dataset = analysis.get("input_dataset")
        if isinstance(global_dataset, dict):
            datasets[(str(global_dataset.get("name", "")), str(global_dataset.get("path", "")))] = global_dataset
        subject = analysis.get("subject", {})
        tasks = subject.get("tasks", {}) if isinstance(subject, dict) else {}
        if isinstance(tasks, dict):
            for task_config in tasks.values():
                if not isinstance(task_config, dict):
                    continue
                dataset = task_config.get("input_dataset")
                if isinstance(dataset, dict):
                    datasets[(str(dataset.get("name", "")), str(dataset.get("path", "")))] = dataset

    invalidate_bids_index("raw", bids_root)
    for name, _path in datasets:
        if name:
            invalidate_bids_index(name, bids_root)

    DatasetIndex.from_config(config)
    for dataset in datasets.values():
        DatasetIndex.from_config(config, task_config={"input_dataset": dataset})


__all__ = [
    "DatasetIndex",
    "get_dataset_index",
    "regenerate_cached_indexes",
]
