from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_icbm152_2009
from nilearn.image import load_img, new_img_like

from ..analysis.analyses.atlas import AtlasCache

logger = logging.getLogger(__name__)

_SUPPORTED_SOURCES = {"", "template_gm"}


def get_group_mask(config: dict[str, Any]):
    """Return the configured reusable group mask image, or ``None`` if disabled."""

    group = config.get("group", {})
    mask_config = group.get("mask", {}) if isinstance(group, dict) else {}
    if not isinstance(mask_config, dict):
        raise ValueError("group.mask must be an object.")

    source = str(mask_config.get("source", "")).strip().lower()
    if source == "":
        return None
    if source not in _SUPPORTED_SOURCES:
        raise ValueError(
            f"Unsupported group mask source {source!r}; expected template_gm."
        )
    return _template_gm_mask(config, mask_config)


def _template_gm_mask(config: dict[str, Any], mask_config: dict[str, Any]):
    threshold = float(mask_config.get("gm_probability_threshold", 0.2))
    cache = AtlasCache(mask_config.get("cache_dir"))
    mask_dir = cache.cache_dir / "group_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    threshold_token = f"{threshold:.6g}".replace(".", "p")
    mask_path = mask_dir / f"template_gm_thr-{threshold_token}.nii.gz"
    if mask_path.exists():
        logger.info("Using cached group mask: %s", mask_path)
        return nib.load(str(mask_path))

    template = fetch_icbm152_2009(data_dir=str(cache.cache_dir), verbose=0)
    probability_image = load_img(template.gm)
    mask_image = new_img_like(
        probability_image,
        np.asarray(probability_image.get_fdata() >= threshold, dtype=np.uint8),
    )
    mask_image.to_filename(mask_path)
    logger.info(
        "Generated template gray matter group mask: source=template_gm threshold=%s path=%s",
        threshold,
        mask_path,
    )
    return mask_image


__all__ = ["get_group_mask"]