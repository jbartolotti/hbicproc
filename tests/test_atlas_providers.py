from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from pipeline.processing.analysis.analyses.atlas import Atlas, AtlasCache, AtlasParcel


def _write_image(path: Path, data: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def test_builtin_atlas_providers_use_cached_fetchers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_image(tmp_path / "atlas.nii.gz", np.array([[[0, 1]]], dtype=np.int16))
    calls = []

    class Fetched:
        maps = str(source)
        labels = ["Background", "Left Hippocampus"]

    def harvard(name: str, **kwargs):
        calls.append(("harvard", name, kwargs["data_dir"]))
        return Fetched()

    def aal(**kwargs):
        calls.append(("aal", kwargs["data_dir"]))
        return Fetched()

    monkeypatch.setattr(
        "pipeline.processing.analysis.analyses.atlas.fetch_atlas_harvard_oxford", harvard
    )
    monkeypatch.setattr("pipeline.processing.analysis.analyses.atlas.fetch_atlas_aal", aal)
    cache = AtlasCache(tmp_path / "cache")

    cortical = cache.get("harvard_oxford_cortical")
    subcortical = cache.get("harvard_oxford_subcortical")
    aal_atlas = cache.get("aal")

    assert cortical.metadata[0].parcel_id == 1
    assert cortical.metadata[0].parcel_label == "Left Hippocampus"
    assert Path(subcortical.maps) == source
    assert aal_atlas.metadata[0].parcel_label == "Left Hippocampus"
    assert calls[0][1] == "cort-maxprob-thr25-2mm"
    assert calls[1][1] == "sub-maxprob-thr25-2mm"
    assert calls[2][0] == "aal"


def test_custom_label_atlas_validates_and_exposes_metadata(tmp_path: Path) -> None:
    atlas_path = _write_image(tmp_path / "custom.nii.gz", np.array([[[0, 1, 2]]], dtype=np.int16))
    labels_path = tmp_path / "labels.tsv"
    pd.DataFrame(
        [
            {"parcel_id": 1, "parcel_label": "Left_Hippocampus", "network": "Memory", "hemisphere": "LH"},
            {"parcel_id": 2, "parcel_label": "Right_Hippocampus", "network": "Memory", "hemisphere": "RH"},
        ]
    ).to_csv(labels_path, sep="\t", index=False)
    cache = AtlasCache(tmp_path / "cache")
    cache.register(
        "memory",
        {"type": "custom_label_atlas", "atlas_file": atlas_path, "labels_file": labels_path},
    )

    atlas = cache.get("memory")

    assert [parcel.parcel_id for parcel in atlas.metadata] == [1, 2]
    assert atlas.metadata[1].hemisphere == "RH"


def test_custom_label_atlas_rejects_unlabeled_roi(tmp_path: Path) -> None:
    atlas_path = _write_image(tmp_path / "custom.nii.gz", np.array([[[0, 1, 2]]], dtype=np.int16))
    labels_path = tmp_path / "labels.tsv"
    labels_path.write_text("parcel_id\tparcel_label\n1\tOnly_ROI\n", encoding="utf-8")
    cache = AtlasCache(tmp_path / "cache")
    cache.register(
        "invalid",
        {"type": "custom_label_atlas", "atlas_file": atlas_path, "labels_file": labels_path},
    )

    with pytest.raises(ValueError, match="missing parcel IDs"):
        cache.get("invalid")


def test_coordinate_spheres_participate_in_contrast_extraction(tmp_path: Path) -> None:
    roi_path = tmp_path / "rois.tsv"
    roi_path.write_text(
        "roi_label\tx\ty\tz\tradius_mm\tnetwork\n"
        "Left_DLPFC\t1\t1\t1\t0.1\tExecutive\n",
        encoding="utf-8",
    )
    cache = AtlasCache(tmp_path / "cache")
    cache.register("executive", {"type": "coordinate_spheres", "roi_file": roi_path})
    target = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=float), np.eye(4))
    effect = nib.Nifti1Image(np.ones((3, 3, 3), dtype=float), np.eye(4))
    output = tmp_path / "summary.tsv"

    cache.roi_contrast_summary("executive", {"task": effect}, target, output)

    table = pd.read_csv(output, sep="\t")
    assert table.loc[0, "parcel_label"] == "Left_DLPFC"
    assert table.loc[0, "network"] == "Executive"
    assert table.loc[0, "effect"] == 1.0


def test_in_memory_atlas_maps_are_accepted(tmp_path: Path) -> None:
    atlas = nib.Nifti1Image(np.array([[[1]]], dtype=np.int16), np.eye(4))
    target = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=float), np.eye(4))
    effect = nib.Nifti1Image(np.array([[[2.5]]]), np.eye(4))
    cache = AtlasCache(tmp_path / "cache")
    cache._atlases["memory"] = Atlas(
        name="memory",
        maps=atlas,
        metadata=(AtlasParcel(1, "Parcel", "memory"),),
    )
    output = tmp_path / "memory.tsv"

    cache.roi_contrast_summary("memory", {"contrast": effect}, target, output)

    table = pd.read_csv(output, sep="\t")
    assert table.loc[0, "effect"] == 2.5
