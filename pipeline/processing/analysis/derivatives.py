from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .context import TaskRunContext


def _normalize_entity(value: str | None, *, prefix: str) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    for candidate in (prefix, prefix.replace("-", "_")):
        if text.lower().startswith(candidate.lower()):
            text = text[len(candidate) :]
            break

    text = text.replace("_", "-")
    return text or None


def _normalize_namespace(value: str, *, kind: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{kind} namespace names must not be empty.")
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{kind} namespace names must be single path components: {value!r}.")
    return text


def _normalize_bids_component(value: str, *, kind: str) -> str:
    text = _normalize_namespace(value, kind=kind)
    return text.replace("_", "-").replace(" ", "-")


@dataclass(frozen=True)
class DerivativePaths:
    """Task/run namespace with isolated model and analysis directories."""

    task_directory: Path
    model_directory: Path | None = None
    analysis_directory: Path | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "task_directory": str(self.task_directory),
            "model_directory": str(self.model_directory) if self.model_directory else None,
            "analysis_directory": str(self.analysis_directory) if self.analysis_directory else None,
        }


class DerivativePathBuilder:
    """Build the task, model, and analysis derivative namespaces."""

    @classmethod
    def task_directory(
        cls,
        base_dir: str | Path,
        *,
        subject: str,
        session: str | None = None,
        task: str,
        run: str | None = None,
    ) -> Path:
        subject_value = _normalize_entity(subject, prefix="sub-")
        task_value = _normalize_entity(task, prefix="task-")
        if subject_value is None or task_value is None:
            raise ValueError("Task derivatives require both subject and task entities.")

        path = Path(base_dir) / f"sub-{subject_value}"
        session_value = _normalize_entity(session, prefix="ses-")
        if session_value is not None:
            path = path / f"ses-{session_value}"

        path = path / "func" / f"task-{task_value}"
        run_value = _normalize_entity(run, prefix="run-")
        if run_value is not None:
            path = path / f"run-{run_value}"
        return path

    @classmethod
    def model_directory(
        cls,
        base_dir: str | Path,
        *,
        subject: str,
        task: str,
        model: str,
        session: str | None = None,
        run: str | None = None,
    ) -> Path:
        return cls.task_directory(
            base_dir,
            subject=subject,
            session=session,
            task=task,
            run=run,
        ) / "models" / _normalize_namespace(model, kind="Model")

    @classmethod
    def analysis_directory(
        cls,
        base_dir: str | Path,
        *,
        subject: str,
        task: str,
        analysis: str,
        session: str | None = None,
        run: str | None = None,
    ) -> Path:
        return cls.task_directory(
            base_dir,
            subject=subject,
            session=session,
            task=task,
            run=run,
        ) / "analyses" / _normalize_namespace(analysis, kind="Analysis")

    @classmethod
    def for_context(
        cls,
        base_dir: str | Path,
        context: TaskRunContext,
        *,
        namespace: str,
        name: str,
    ) -> DerivativePaths:
        task_path = cls.task_directory(
            base_dir,
            subject=context.subject,
            session=context.session,
            task=context.task,
            run=context.run,
        )
        if namespace == "model":
            return DerivativePaths(
                task_directory=task_path,
                model_directory=task_path / "models" / _normalize_namespace(name, kind="Model"),
            )
        if namespace == "analysis":
            return DerivativePaths(
                task_directory=task_path,
                analysis_directory=task_path / "analyses" / _normalize_namespace(name, kind="Analysis"),
            )
        raise ValueError(f"Unknown derivative namespace: {namespace!r}.")

    @classmethod
    def build_file(
        cls,
        base_dir: str | Path,
        *,
        subject: str,
        session: str | None = None,
        task: str,
        run: str | None = None,
        namespace: str,
        namespace_name: str,
        desc: str | None = None,
        stat: str | None = None,
        suffix: str = ".nii.gz",
    ) -> Path:
        if namespace == "model":
            directory = cls.model_directory(
                base_dir,
                subject=subject,
                session=session,
                task=task,
                run=run,
                model=namespace_name,
            )
        elif namespace == "analysis":
            directory = cls.analysis_directory(
                base_dir,
                subject=subject,
                session=session,
                task=task,
                run=run,
                analysis=namespace_name,
            )
        else:
            raise ValueError(f"Unknown derivative namespace: {namespace!r}.")

        return directory / cls.build_filename(
            subject=subject,
            session=session,
            task=task,
            run=run,
            desc=desc,
            stat=stat,
            suffix=suffix,
        )

    @staticmethod
    def build_filename(
        *,
        subject: str,
        session: str | None = None,
        task: str,
        run: str | None = None,
        desc: str | None = None,
        stat: str | None = None,
        suffix: str = ".nii.gz",
    ) -> str:
        entity_tokens = [
            f"sub-{_normalize_entity(subject, prefix='sub-')}",
        ]
        session_value = _normalize_entity(session, prefix="ses-")
        if session_value is not None:
            entity_tokens.append(f"ses-{session_value}")
        entity_tokens.append(f"task-{_normalize_entity(task, prefix='task-')}")
        run_value = _normalize_entity(run, prefix="run-")
        if run_value is not None:
            entity_tokens.append(f"run-{run_value}")

        tokens = [f"desc-{_normalize_bids_component(desc, kind='Description')}"] if desc else []
        if stat:
            tokens.append(f"stat-{_normalize_bids_component(stat, kind='Statistic')}")
        stem = "_".join(entity_tokens + tokens) if tokens else "_".join(entity_tokens)

        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return f"{stem}{normalized_suffix}"


__all__ = ["DerivativePaths", "DerivativePathBuilder"]
