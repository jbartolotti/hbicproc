from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_icbm152_2009
from nilearn.image import load_img, new_img_like, resample_to_img

from ..analysis.analyses.atlas import AtlasCache

logger = logging.getLogger(__name__)

_SUPPORTED_SOURCES = {"", "template_gm"}


def get_group_mask(config: dict[str, Any], reference_image: str | Path | Any | None = None):
    """Return a configured group mask aligned to the supplied study image grid."""

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
    if reference_image is None:
        raise ValueError("A reference image is required when group.mask.source is 'template_gm'.")
    return _template_gm_mask(mask_config, reference_image)


def _template_gm_mask(mask_config: dict[str, Any], reference_image: str | Path | Any):
    threshold = float(mask_config.get("gm_probability_threshold", 0.2))
    cache = AtlasCache(mask_config.get("cache_dir"))
    mask_dir = cache.cache_dir / "group_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    target_image = load_img(reference_image)
    target_shape = tuple(int(value) for value in target_image.shape[:3])
    target_voxel_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(target_image.affine)[:3])
    target_grid = np.asarray(target_image.affine, dtype=np.float64).tobytes() + str(target_shape).encode()
    grid_hash = hashlib.sha256(target_grid).hexdigest()[:12]
    threshold_token = f"{threshold:.6g}".replace(".", "p")
    mask_path = mask_dir / f"template_gm_thr-{threshold_token}_grid-{grid_hash}.nii.gz"
    if mask_path.exists():
        logger.info("Using cached resampled group mask: %s", mask_path)
        return nib.load(str(mask_path))

    template = fetch_icbm152_2009(data_dir=str(cache.cache_dir), verbose=0)
    probability_image = load_img(template.gm)
    template_mask = new_img_like(
        probability_image,
        np.asarray(probability_image.get_fdata() >= threshold, dtype=np.uint8),
    )
    logger.info(
        "Resampling group mask: original_shape=%s original_voxel_size=%s "
        "target_shape=%s target_voxel_size=%s",
        template_mask.shape,
        tuple(float(value) for value in nib.affines.voxel_sizes(template_mask.affine)[:3]),
        target_shape,
        target_voxel_sizes,
    )
    mask_image = resample_to_img(
        template_mask,
        target_image,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    mask_image = new_img_like(
        target_image,
        np.asarray(mask_image.get_fdata() >= 0.5, dtype=np.uint8),
    )
    mask_image.to_filename(mask_path)
    logger.info(
        "Generated resampled template gray matter group mask: source=template_gm "
        "threshold=%s cached_path=%s",
        threshold,
        mask_path,
    )
    return mask_image


__all__ = ["get_group_mask"]