from __future__ import annotations

from pathlib import Path


class DerivativePathBuilder:
    """Build BIDS-derivatives paths with exact subject/session/task/run entities."""

    @staticmethod
    def _normalize_entity(value: str | None, *, prefix: str | None = None) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        if prefix and text.startswith(prefix):
            text = text[len(prefix) :]

        normalized = text.replace("_", "-")
        if not normalized:
            return None
        return normalized

    @classmethod
    def build(
        cls,
        base_dir: str | Path,
        *,
        subject: str | None = None,
        session: str | None = None,
        task: str | None = None,
        run: str | None = None,
        desc: str | None = None,
        stat: str | None = None,
        suffix: str = ".nii.gz",
    ) -> Path:
        tokens: list[str] = []

        subject_value = cls._normalize_entity(subject, prefix="sub-")
        if subject_value:
            tokens.append(f"sub-{subject_value}")

        session_value = cls._normalize_entity(session, prefix="ses-")
        if session_value:
            tokens.append(f"ses-{session_value}")

        task_value = cls._normalize_entity(task, prefix="task-")
        if task_value:
            tokens.append(f"task-{task_value}")

        run_value = cls._normalize_entity(run, prefix="run-")
        if run_value:
            tokens.append(f"run-{run_value}")

        desc_value = cls._normalize_entity(desc, prefix="desc-")
        if desc_value is not None:
            tokens.append(f"desc-{desc_value}")

        stat_value = cls._normalize_entity(stat, prefix="stat-")
        if stat_value is not None:
            tokens.append(f"stat-{stat_value}")

        stem = "_".join(tokens)
        if not stem:
            raise ValueError("Derivative output requires at least a subject entity.")

        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return Path(base_dir) / f"{stem}{normalized_suffix}"
