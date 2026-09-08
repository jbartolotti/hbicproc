from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


_ENTITY_RE = re.compile(r"(?:^|_)sub-([^_]+)|(?:^|_)ses-([^_]+)")


def _bids_component(value: str) -> str:
    return str(value).strip().replace("_", "-").replace(" ", "-")


def _entities(path: Path) -> tuple[str | None, str | None]:
    text = path.name
    subject = None
    session = None
    for match in _ENTITY_RE.finditer(text):
        if match.group(1):
            subject = match.group(1)
        if match.group(2):
            session = match.group(2)
    if subject is None:
        for part in path.parts:
            if part.startswith("sub-"):
                subject = part[4:]
            elif part.startswith("ses-"):
                session = part[4:]
    return subject, session


def discover_effect_maps(
    derivatives_root: str | Path,
    *,
    task: str,
    contrast: str,
) -> pd.DataFrame:
    """Discover existing subject-level effect maps for one configured contrast."""

    root = Path(derivatives_root)
    task_component = _bids_component(task)
    contrast_component = _bids_component(contrast)
    pattern = f"**/*_task-{task_component}*_desc-{contrast_component}_stat-effect.nii.gz"
    records = []
    for path in sorted(root.glob(pattern)):
        subject, session = _entities(path)
        if subject is not None:
            records.append({"subject": subject, "session": session, "path": path})
    return pd.DataFrame(records, columns=["subject", "session", "path"])


def discover_atlas_activation_summaries(
    derivatives_root: str | Path,
    *,
    task: str,
    atlas: str,
) -> pd.DataFrame:
    """Discover atlas activation summary TSVs for one configured task and atlas."""

    root = Path(derivatives_root)
    task_component = _bids_component(task)
    atlas_component = _bids_component(atlas)
    pattern = f"**/*_task-{task_component}*_desc-atlas-{atlas_component}-activation.tsv"
    records = []
    for path in sorted(root.glob(pattern)):
        subject, session = _entities(path)
        if subject is not None:
            table = pd.read_csv(path, sep="\t")
            table["subject"] = subject
            table["session"] = session
            table["source_path"] = str(path)
            records.append(table)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)
