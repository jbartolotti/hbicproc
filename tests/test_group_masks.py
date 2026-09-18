from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pipeline.config import _apply_defaults, validate_config
from pipeline.processing.group import masks


def test_group_mask_configuration_defaults_and_validates() -> None:
    config = _apply_defaults({})
    validate_config(config)

    assert config["group"]["mask"] == {
        "source": "",
        "gm_probability_threshold": 0.2,
    }

    invalid = _apply_defaults({"group": {"mask": {"source": "group_gm"}}})
    with pytest.raises(ValueError, match="group.mask.source"):
        validate_config(invalid)


def test_template_gm_mask_is_thresholded_and_cached(tmp_path: Path, monkeypatch) -> None:
    probability_path = tmp_path / "gm.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.array([[[0.1, 0.2], [0.7, 0.0]]], dtype=float), np.eye(4)
        ),
        probability_path,
    )
    fetch_calls = []

    class Fetched:
        gm = str(probability_path)

    def fetch_template(**kwargs):
        fetch_calls.append(kwargs)
        return Fetched()

    monkeypatch.setattr(masks, "fetch_icbm152_2009", fetch_template)
    config = {
        "group": {
            "mask": {
                "source": "template_gm",
                "gm_probability_threshold": 0.2,
                "cache_dir": str(tmp_path / "cache"),
            }
        }
    }

    first = masks.get_group_mask(config)
    second = masks.get_group_mask(config)

    assert fetch_calls[0]["data_dir"] == str(tmp_path / "cache")
    assert len(fetch_calls) == 1
    assert np.array_equal(first.get_fdata(), second.get_fdata())
    assert np.array_equal(first.get_fdata(), np.array([[[0, 1], [1, 0]]], dtype=float))
    assert set(np.unique(first.get_fdata())) == {0.0, 1.0}