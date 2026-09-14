from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from .context import TaskRunContext


_MASK_STRATEGIES = {"derivative", "nilearn", "explicit"}


@dataclass(frozen=True)
class MaskResolution:
    strategy: str
    source: str
    mask_file: Path | None
    mask_found: bool
    fallback_used: bool
    resampled: bool
    mask_img: Any | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mask_strategy": self.strategy,
            "mask_source": self.source,
            "mask_file": str(self.mask_file) if self.mask_file is not None else None,
            "mask_found": self.mask_found,
            "fallback_used": self.fallback_used,
            "resampled": self.resampled,
        }


class MaskResolutionError(ValueError):
    """Raised when an explicitly requested mask cannot be used."""


def resolve_mask(
    context: TaskRunContext,
    configuration: dict[str, Any],
    *,
    logger: Any,
) -> MaskResolution:
    strategy = str(configuration.get("mask_strategy", "derivative")).strip().lower()
    if strategy not in _MASK_STRATEGIES:
        raise MaskResolutionError(
            f"Unsupported mask_strategy {strategy!r}; expected derivative, nilearn, or explicit."
        )

    if strategy == "nilearn":
        logger.info("Using Nilearn automatic mask estimation for %s", context.identity)
        return MaskResolution(strategy, "nilearn_estimated", None, False, False, False, None)

    if strategy == "derivative":
        mask_path = context.derivative_mask_path
        if mask_path is None or not mask_path.exists():
            return _fallback_or_raise(
                context,
                configuration,
                strategy,
                "No derivative mask found for %s; falling back to Nilearn mask estimation.",
                logger=logger,
            )
        try:
            mask_img, resampled = _load_compatible_mask(mask_path, context.bold_path)
        except MaskResolutionError as exc:
            return _fallback_or_raise(
                context,
                configuration,
                strategy,
                "Derivative mask %s is unusable for %s (%s); falling back to Nilearn mask estimation.",
                logger=logger,
                error=exc,
                mask_path=mask_path,
            )
        logger.info("Using derivative mask: %s", mask_path)
        if resampled:
            logger.info("Mask resampled to match target imaging space: %s", context.bold_path)
        return MaskResolution(strategy, "derivative_mask", mask_path, True, False, resampled, mask_img)

    template = str(configuration.get("mask_path_template", "")).strip()
    if not template:
        raise MaskResolutionError(
            "Explicit mask strategy requires a non-empty mask_path_template."
        )
    mask_path = _resolve_template(template, context)
    if not mask_path.exists():
        raise MaskResolutionError(
            f"Explicit mask was not found for {context.identity}: {mask_path}"
        )
    mask_img, resampled = _load_compatible_mask(mask_path, context.bold_path)
    logger.info("Using explicit mask: %s", mask_path)
    if resampled:
        logger.info("Mask resampled to match target imaging space: %s", context.bold_path)
    return MaskResolution(strategy, "explicit_mask", mask_path, True, False, resampled, mask_img)


def _fallback_or_raise(
    context: TaskRunContext,
    configuration: dict[str, Any],
    strategy: str,
    message: str,
    *,
    logger: Any,
    error: Exception | None = None,
    mask_path: Path | None = None,
) -> MaskResolution:
    if not bool(configuration.get("nilearn_fallback_mask", True)):
        detail = f" ({error})" if error is not None else ""
        path_detail = f": {mask_path}" if mask_path is not None else ""
        raise MaskResolutionError(
            f"Requested derivative mask is unavailable{path_detail} for {context.identity}{detail}; "
            "set nilearn_fallback_mask=true to permit Nilearn fallback."
        ) from error
    if error is None:
        logger.warning(message, context.identity)
    else:
        logger.warning(message, mask_path, context.identity, error)
    return MaskResolution(strategy, "nilearn_estimated", None, False, True, False, None)


def _resolve_template(template: str, context: TaskRunContext) -> Path:
    values = {
        "subject": context.subject,
        "session": context.session or "",
        "task": context.task,
        "run": context.run or "",
    }
    try:
        return Path(template.format(**values))
    except KeyError as exc:
        raise MaskResolutionError(
            f"Explicit mask_path_template contains unsupported placeholder {exc.args[0]!r}."
        ) from exc


def _load_compatible_mask(mask_path: Path, bold_path: Path | None) -> tuple[Any, bool]:
    if bold_path is None:
        raise MaskResolutionError("Mask compatibility cannot be checked without a BOLD image.")
    mask_img = nib.load(str(mask_path))
    bold_img = nib.load(str(bold_path))
    _validate_binary_mask(mask_img, mask_path)
    if mask_img.shape == bold_img.shape[:3] and np.allclose(mask_img.affine, bold_img.affine):
        return mask_img, False
    try:
        resampled = resample_to_img(
            mask_img,
            bold_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    except Exception as exc:
        raise MaskResolutionError(
            f"could not resample mask from shape {mask_img.shape} and affine {mask_img.affine.tolist()} "
            f"to BOLD shape {bold_img.shape[:3]} and affine {bold_img.affine.tolist()}: {exc}"
        ) from exc
    _validate_binary_mask(resampled, mask_path)
    return resampled, True


def _validate_binary_mask(image: Any, path: Path) -> None:
    if len(image.shape) != 3:
        raise MaskResolutionError(f"Mask must be 3D: {path} has shape {image.shape}.")
    data = np.asarray(image.get_fdata())
    if not np.isfinite(data).all():
        raise MaskResolutionError(f"Mask contains non-finite values: {path}.")
    values = np.unique(data)
    if values.size > 2 or not np.all(np.isclose(values, 0) | np.isclose(values, 1)):
        raise MaskResolutionError(
            f"Mask must contain only binary 0/1 values: {path} contains {values[:8].tolist()}."
        )
    if not np.any(data > 0):
        raise MaskResolutionError(f"Mask is empty: {path}.")


__all__ = ["MaskResolution", "MaskResolutionError", "resolve_mask"]
