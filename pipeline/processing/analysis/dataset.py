from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bids import BIDSLayout, BIDSLayoutIndexer

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


@dataclass(frozen=True)
class AnalysisRun:
    subject: str
    task: str
    session: str | None = None
    run: str | None = None
    bold_path: Path | str = ""
    events_path: Path | str = ""
    confounds_path: Path | str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "task": self.task,
            "session": self.session,
            "run": self.run,
            "bold_path": str(self.bold_path),
            "events_path": str(self.events_path),
            "confounds_path": str(self.confounds_path),
        }


class DatasetIndex:
    """Central BIDS-aware dataset index for discovery-oriented analysis plugins."""

    def __init__(
        self,
        bids_root: str | Path | None = None,
        *,
        study_root: str | Path | None = None,
        derivative_dataset: str | Path = "fmriprep",
    ) -> None:
        root = Path(bids_root) if bids_root is not None else (Path(study_root) if study_root is not None else Path("."))
        self.bids_root = root.resolve() if root.exists() else root
        self.study_root = Path(study_root).resolve() if study_root is not None and Path(study_root).exists() else self.bids_root
        self.derivative_dataset = str(derivative_dataset)
        derivative_path = Path(derivative_dataset)
        if not derivative_path.is_absolute():
            derivative_path = self.bids_root / "derivatives" / derivative_path
        self.derivative_root = derivative_path.resolve() if derivative_path.exists() else derivative_path

        self.layout: BIDSLayout | None = None
        self.derivative_layout: BIDSLayout | None = None
        self._task_runs_cache: dict[
            tuple[str | None, str | None, str | None, str | None],
            tuple[AnalysisRun, ...],
        ] = {}
        if self.bids_root.exists():
            try:
                self.layout = BIDSLayout(
                    str(self.bids_root),
                    validate=False,
                    derivatives=False,
                    indexer=BIDSLayoutIndexer(validate=False),
                )
            except Exception:
                self.layout = None
        if self.derivative_root.exists():
            try:
                self.derivative_layout = BIDSLayout(
                    str(self.derivative_root),
                    validate=False,
                    derivatives=False,
                    is_derivative=True,
                    indexer=BIDSLayoutIndexer(validate=False),
                )
            except Exception:
                self.derivative_layout = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DatasetIndex":
        bids_root = config.get("bids_root") or config.get("study_root") or "."
        study_root = config.get("study_root") or bids_root
        analysis = config.get("analysis", {})
        derivative_dataset = "fmriprep"
        if isinstance(analysis, dict):
            derivative_dataset = analysis.get("derivative_dataset") or derivative_dataset
        return cls(bids_root=bids_root, study_root=study_root, derivative_dataset=derivative_dataset)

    def get_task_runs(
        self,
        *,
        subject: str | None = None,
        task: str | None = None,
        session: str | None = None,
        run: str | None = None,
    ) -> list[AnalysisRun]:
        normalized_subject = _normalize_entity_text(subject, prefix="sub") if subject is not None else None
        normalized_task = _normalize_entity_text(task, prefix="task") if task is not None else None
        normalized_session = _normalize_entity_text(session, prefix="ses") if session is not None else None
        normalized_run = _normalize_entity_text(run, prefix="run") if run is not None else None
        cache_key = (normalized_subject, normalized_task, normalized_session, normalized_run)
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

        if self.layout is None or self.derivative_layout is None:
            logger.info(
                "Dataset discovery skipped for subject=%s task=%s: raw_layout=%s derivative_layout=%s derivative_dataset=%s derivative_root=%s",
                subject,
                task,
                self.layout is not None,
                self.derivative_layout is not None,
                self.derivative_dataset,
                self.derivative_root,
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
            self.derivative_layout,
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
            self.derivative_layout,
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

        runs: list[AnalysisRun] = []
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

            resolved_run = AnalysisRun(
                subject=subject_value or "",
                task=task_value or "",
                session=session_value,
                run=run_value,
                bold_path=Path(str(bold_record.path)),
                events_path=Path(str(event_match.path)),
                confounds_path=Path(str(confounds_match.path)),
            )
            runs.append(resolved_run)

        logger.info("Resolved AnalysisRun objects for subject=%s task=%s: %s", normalized_subject, normalized_task, [run.as_dict() for run in runs])
        resolved_runs = tuple(sorted(runs, key=lambda item: (item.session or "", item.run or "")))
        self._task_runs_cache[cache_key] = resolved_runs
        return list(resolved_runs)

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


__all__ = [
    "AnalysisRun",
    "DatasetIndex",
    "get_dataset_index",
]
