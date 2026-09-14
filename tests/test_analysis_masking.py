from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pipeline.config import load_default_config
from pipeline.processing.analysis.context import TaskRunContext
from pipeline.processing.analysis.masking import MaskResolutionError, resolve_mask


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(("info", message, args))

    def warning(self, message, *args):
        self.messages.append(("warning", message, args))


def _write_image(path: Path, shape=(4, 4, 4), affine=None, value=1):
    affine = np.eye(4) if affine is None else affine
    data = np.zeros(shape, dtype=np.uint8)
    data[1:-1, 1:-1, 1:-1] = value
    nib.save(nib.Nifti1Image(data, affine), path)


def _context(tmp_path: Path, mask_path: Path | None = None) -> TaskRunContext:
    bold_path = tmp_path / "bold.nii.gz"
    _write_image(bold_path, shape=(4, 4, 4))
    return TaskRunContext(
        subject="001",
        session="baseline",
        task="nback",
        run="1",
        bold_path=bold_path,
        derivative_mask_path=mask_path,
    )


def test_default_first_level_configuration_uses_derivative_and_fallback() -> None:
    config = load_default_config(root_dir=".")

    assert config["analysis"]["first_level"] == {
        "mask_strategy": "derivative",
        "nilearn_fallback_mask": True,
        "mask_path_template": "",
    }


def test_derivative_mask_is_selected_and_validated(tmp_path: Path) -> None:
    mask_path = tmp_path / "sub-001_task-nback_desc-brain_mask.nii.gz"
    _write_image(mask_path)
    logger = _Logger()

    resolution = resolve_mask(
        _context(tmp_path, mask_path),
        {"mask_strategy": "derivative", "nilearn_fallback_mask": True},
        logger=logger,
    )

    assert resolution.source == "derivative_mask"
    assert resolution.mask_file == mask_path
    assert resolution.mask_found is True
    assert resolution.fallback_used is False
    assert resolution.mask_img.shape == (4, 4, 4)


def test_derivative_missing_mask_can_fail_without_nilearn_fallback(tmp_path: Path) -> None:
    logger = _Logger()

    with pytest.raises(MaskResolutionError, match="nilearn_fallback_mask=true"):
        resolve_mask(
            _context(tmp_path),
            {"mask_strategy": "derivative", "nilearn_fallback_mask": False},
            logger=logger,
        )


def test_derivative_missing_mask_falls_back_by_default(tmp_path: Path) -> None:
    resolution = resolve_mask(
        _context(tmp_path),
        {"mask_strategy": "derivative"},
        logger=_Logger(),
    )

    assert resolution.source == "nilearn_estimated"
    assert resolution.fallback_used is True
    assert resolution.mask_img is None


def test_explicit_mask_template_resolves_run_entities(tmp_path: Path) -> None:
    mask_path = tmp_path / "manual" / "sub-001" / "ses-baseline" / "task-nback_run-1.nii.gz"
    mask_path.parent.mkdir(parents=True)
    _write_image(mask_path)
    template = str(tmp_path / "manual" / "sub-{subject}" / "ses-{session}" / "task-{task}_run-{run}.nii.gz")

    resolution = resolve_mask(
        _context(tmp_path),
        {"mask_strategy": "explicit", "mask_path_template": template},
        logger=_Logger(),
    )

    assert resolution.source == "explicit_mask"
    assert resolution.mask_file == mask_path
    assert resolution.fallback_used is False


def test_explicit_missing_mask_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(MaskResolutionError, match="Explicit mask was not found"):
        resolve_mask(
            _context(tmp_path),
            {
                "mask_strategy": "explicit",
                "mask_path_template": str(tmp_path / "missing-{subject}.nii.gz"),
            },
            logger=_Logger(),
        )


def test_mask_on_different_grid_is_resampled_with_nearest_neighbor(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.nii.gz"
    _write_image(mask_path, shape=(3, 3, 3), affine=np.diag([2, 2, 2, 1]))
    logger = _Logger()

    resolution = resolve_mask(
        _context(tmp_path, mask_path),
        {"mask_strategy": "derivative"},
        logger=logger,
    )

    assert resolution.resampled is True
    assert resolution.mask_img.shape == (4, 4, 4)
    assert any("resampled" in message for _, message, _ in logger.messages)
