from __future__ import annotations

from typing import Type

from .base import QCReport


REPORT_CLASSES: dict[str, Type[QCReport]] = {}


def register_report(report_class: Type[QCReport]) -> Type[QCReport]:
    name = str(report_class.name).strip()
    if not name:
        raise ValueError("QC report names must not be empty.")
    if name in REPORT_CLASSES:
        raise ValueError(f"QC report '{name}' is already registered.")
    REPORT_CLASSES[name] = report_class
    return report_class


def get_report_classes() -> dict[str, Type[QCReport]]:
    from . import motion_qc  # noqa: F401

    return dict(REPORT_CLASSES)