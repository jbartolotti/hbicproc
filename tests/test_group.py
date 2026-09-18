from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from pipeline.processing.group.activation import run_activation
from pipeline.processing.group.atlas_metadata import AtlasMetadata
from pipeline.processing.group.discovery import discover_atlas_contrast_summaries, discover_effect_maps
from pipeline.processing.group.inference import InferenceResult, create_inference_method
from pipeline.processing.group.one_sample import _montages, run_one_sample
from pipeline.processing.group.participants import add_factor_columns, load_participants
from pipeline.config import _apply_defaults, validate_config


def test_group_derivative_discovery_and_participant_session_join(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    derivative_root = tmp_path / "derivatives" / "hbicproc"
    bids_root.mkdir(parents=True)
    map_path = (
        derivative_root / "sub-001" / "ses-BL" / "func" / "task-nback" / "analyses" / "activation"
        / "sub-001_ses-BL_task-nback_desc-2back-gt-1back_stat-effect.nii.gz"
    )
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"fake")
    (bids_root / "participants.tsv").write_text(
        "participant_id\tgroup\nsub-001\tintervention\n", encoding="utf-8"
    )

    records = discover_effect_maps(derivative_root, task="nback", contrast="2back_gt_1back")
    participants = load_participants(bids_root)
    joined = add_factor_columns(
        records,
        participants,
        {
            "group": {"source": "participants.tsv", "column": "group"},
            "session": {"source": "bids_session", "levels": {"BL": "baseline"}},
        },
    )

    assert joined.loc[0, "group"] == "intervention"
    assert joined.loc[0, "session"] == "baseline"
    assert joined.loc[0, "path"] == map_path


def test_atlas_contrast_summary_discovery_preserves_entities_and_filters_contrast(tmp_path: Path) -> None:
    path = (
        tmp_path / "sub-001" / "ses-W12" / "func" / "task-nback" / "analyses" / "activation"
        / "sub-001_ses-W12_task-nback_desc-atlas-schaefer200-contrasts.tsv"
    )
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"parcel_id": 1, "parcel_label": "parcel", "network": "Default", "hemisphere": "LH", "contrast": "memory", "effect": 1.5},
            {"parcel_id": 1, "parcel_label": "parcel", "network": "Default", "hemisphere": "LH", "contrast": "other", "effect": 2.5},
        ]
    ).to_csv(path, sep="\t", index=False)

    records = discover_atlas_contrast_summaries(
        tmp_path, task="nback", atlas="schaefer200", contrast="memory"
    )

    assert len(records) == 1
    assert records.loc[0, "subject"] == "001"
    assert records.loc[0, "session"] == "W12"
    assert records.loc[0, "effect"] == 1.5
    assert records.loc[0, "source_path"] == str(path)


def test_atlas_metadata_selects_network_without_parcel_numbers() -> None:
    class FakeAtlas:
        labels = ("7Networks_LH_Default_PFC_1", "7Networks_RH_Vis_Visual_1")

    class FakeCache:
        def get(self, name: str) -> FakeAtlas:
            assert name == "schaefer200"
            return FakeAtlas()

    metadata = AtlasMetadata.from_cache(FakeCache(), "schaefer200")
    parcels = metadata.get_parcels(network="Default")

    assert len(parcels) == 1
    assert parcels[0].parcel_label == "7Networks_LH_Default_PFC_1"
    assert parcels[0].hemisphere == "LH"


def test_group_activation_fits_configured_effect_maps(tmp_path: Path, monkeypatch) -> None:
    class FakeImage:
        def to_filename(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"map")

    class FakeSecondLevelModel:
        def __init__(self, **kwargs):
            self.design_matrix = None

        def fit(self, images, design_matrix):
            self.design_matrix = design_matrix
            return self

        def compute_contrast(self, vector, output_type):
            return FakeImage()

    monkeypatch.setattr(
        "pipeline.processing.group.activation.SecondLevelModel",
        FakeSecondLevelModel,
    )
    bids_root = tmp_path / "bids"
    derivative_root = tmp_path / "derivatives" / "hbicproc"
    (bids_root / "participants.tsv").parent.mkdir(parents=True)
    (bids_root / "participants.tsv").write_text(
        "participant_id\tgroup\nsub-001\tcontrol\nsub-002\tcontrol\n"
        "sub-003\tintervention\nsub-004\tintervention\n",
        encoding="utf-8",
    )
    for subject, group, session in [
        ("001", "control", "BL"), ("002", "control", "W12"),
        ("003", "intervention", "BL"), ("004", "intervention", "W12"),
    ]:
        path = (
            derivative_root / f"sub-{subject}" / f"ses-{session}" / "func" / "task-nback"
            / "analyses" / "activation"
            / f"sub-{subject}_ses-{session}_task-nback_desc-2back-gt-1back_stat-effect.nii.gz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    result = run_activation(
        {
            "bids_root": str(bids_root),
            "analysis": {"output_dir": str(derivative_root)},
        },
        {
            "enabled": True,
            "task": "nback",
            "contrasts": ["2back_gt_1back"],
            "factors": {
                "group": {"source": "participants.tsv", "column": "group"},
                "session": {"source": "bids_session", "levels": {"BL": "baseline", "W12": "followup"}},
            },
        },
    )

    assert result["results"][0]["n_maps"] == 4
    design = pd.read_csv(
        derivative_root / "group" / "activation" / "2back-gt-1back" / "design_matrix.tsv",
        sep="\t",
    )
    assert list(design.columns) == ["intercept", "group", "session", "group_session"]


def test_one_sample_runs_each_session_and_respects_session_filter(tmp_path: Path, monkeypatch) -> None:
    bids_root = tmp_path / "bids"
    derivative_root = tmp_path / "derivatives" / "hbicproc"
    bids_root.mkdir(parents=True)
    (bids_root / "participants.tsv").write_text(
        "participant_id\nsub-001\nsub-002\n", encoding="utf-8"
    )
    for session in ("BL", "w12"):
        for subject in ("001", "002"):
            path = (
                derivative_root / f"sub-{subject}" / f"ses-{session}" / "func" / "task-nback"
                / "analyses" / "activation"
                / f"sub-{subject}_ses-{session}_task-nback_desc-memory_stat-effect.nii.gz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=float), np.eye(4)), path)

    class FakeImage:
        def __init__(self, value: float):
            self.value = value

        def to_filename(self, path: Path) -> None:
            data = np.zeros((3, 3, 3), dtype=float)
            data[1, 1, 1] = self.value
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    class FakeSecondLevelModel:
        fit_count = 0

        def __init__(self, **kwargs):
            self.design_matrix = None

        def fit(self, images, design_matrix):
            self.design_matrix = design_matrix
            FakeSecondLevelModel.fit_count += 1
            return self

        def compute_contrast(self, vector, output_type):
            return FakeImage(2.0 if output_type == "effect_size" else 4.0)

    class FakeViewer:
        def save_as_html(self, path: str) -> None:
            Path(path).write_text("<html>viewer</html>", encoding="utf-8")

    monkeypatch.setattr("pipeline.processing.group.one_sample.SecondLevelModel", FakeSecondLevelModel)
    monkeypatch.setattr("pipeline.processing.group.one_sample.view_img", lambda *args, **kwargs: FakeViewer())
    monkeypatch.setattr(
        "pipeline.processing.group.one_sample._montages",
        lambda *args: (args[1].write_bytes(b"png"), args[2].write_bytes(b"png")),
    )

    config = {
        "bids_root": str(bids_root),
        "study_root": str(tmp_path),
        "analysis": {"output_dir": str(derivative_root)},
    }
    specification = {
        "enabled": True,
        "task": "nback",
        "contrasts": ["memory"],
        "inference": {"method": "fdr", "alpha": 0.05},
    }

    result = run_one_sample(config, specification)

    assert FakeSecondLevelModel.fit_count == 2
    assert {(row["session"], row["n_subjects"]) for row in result["results"]} == {
        ("BL", 2),
        ("w12", 2),
    }
    assert (derivative_root / "group" / "one_sample_group_report.html").exists()
    assert len(list((derivative_root / "group" / "one_sample").rglob("viewer.html"))) == 2

    filtered = run_one_sample(config, {**specification, "sessions": ["BL"]})

    assert len(filtered["results"]) == 1
    assert filtered["results"][0]["session"] == "BL"

    no_clusters = run_one_sample(
        config, {**specification, "sessions": ["BL"], "inference": {"method": "fdr", "alpha": 1e-6}}
    )
    assert no_clusters["results"][0]["n_clusters"] == 0
    assert "No clusters survived" in (derivative_root / "group" / "one_sample_group_report.html").read_text(encoding="utf-8")


def test_one_sample_rejects_multiple_tasks_without_task_filter(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    derivative_root = tmp_path / "derivatives" / "hbicproc"
    bids_root.mkdir(parents=True)
    (bids_root / "participants.tsv").write_text("participant_id\nsub-001\n", encoding="utf-8")
    for task in ("nback", "stroop"):
        path = derivative_root / "sub-001" / "func" / f"task-{task}" / "analyses" / "activation" / f"sub-001_task-{task}_desc-memory_stat-effect.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=float), np.eye(4)), path)

    with pytest.raises(ValueError, match="multiple tasks"):
        run_one_sample(
            {"bids_root": str(bids_root), "analysis": {"output_dir": str(derivative_root)}},
            {"contrasts": ["memory"], "inference": {"method": "fdr", "alpha": 0.05}},
        )


def test_one_sample_configuration_validates_session_and_cluster_settings() -> None:
    config = _apply_defaults({
        "group": {
            "one_sample": {
                "enabled": True,
                "contrasts": ["memory"],
                "sessions": ["BL"],
                "inference": {"method": "fdr", "alpha": 0.05},
            }
        }
    })
    validate_config(config)

    invalid = _apply_defaults({
        "group": {
            "one_sample": {
                "enabled": True,
                "contrasts": ["memory"],
                "sessions": [""],
                "inference": {"method": "fdr", "alpha": 0.05},
            }
        }
    })
    with pytest.raises(ValueError, match="sessions"):
        validate_config(invalid)

    invalid["group"]["one_sample"]["sessions"] = ["BL"]
    invalid["group"]["one_sample"]["inference"]["alpha"] = 0
    with pytest.raises(ValueError, match="alpha"):
        validate_config(invalid)

    invalid["group"]["one_sample"]["inference"]["alpha"] = 0.05
    invalid["group"]["one_sample"]["inference"]["method"] = "permutation_cluster"
    with pytest.raises(ValueError, match="method"):
        validate_config(invalid)


def test_fdr_inference_returns_common_result_products(tmp_path: Path) -> None:
    stat_map = tmp_path / "group_stat-z.nii.gz"
    data = np.zeros((5, 5, 5), dtype=float)
    data[2, 2, 2] = 8.0
    nib.save(nib.Nifti1Image(data, np.eye(4)), stat_map)

    result = create_inference_method({"method": "fdr"}).run(
        stat_map, tmp_path / "inference", {"method": "fdr", "alpha": 0.05}
    )

    assert result.stat_map == stat_map
    assert result.thresholded_map.exists()
    assert result.significance_mask.exists()
    assert result.cluster_table.exists()
    assert result.metadata["method"] == "fdr"
    assert result.metadata["alpha"] == 0.05
    assert "computed_threshold" in result.metadata


def test_static_montages_use_nilearn_symmetric_colorbar_api(tmp_path: Path, monkeypatch) -> None:
    shape = (4, 4, 4)
    stat_map = tmp_path / "stat.nii.gz"
    thresholded_map = tmp_path / "thresholded.nii.gz"
    significance_mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), stat_map)
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), thresholded_map)
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), significance_mask)
    cluster_table = tmp_path / "clusters.tsv"
    cluster_table.write_text("cluster_index\n1\n", encoding="utf-8")
    inference = InferenceResult(
        stat_map=stat_map,
        thresholded_map=thresholded_map,
        significance_mask=significance_mask,
        cluster_table=cluster_table,
        metadata={"method": "fdr"},
    )
    calls = []

    class FakeDisplay:
        def add_contours(self, *args, **kwargs):
            return None

    def fake_plot_stat_map(*args, **kwargs):
        calls.append(kwargs)
        return FakeDisplay()

    monkeypatch.setattr("pipeline.processing.group.one_sample.plot_stat_map", fake_plot_stat_map)
    _montages(inference, tmp_path / "thresholded.png", tmp_path / "unthresholded.png", "test")

    assert len(calls) == 2
    assert all(call["symmetric_cbar"] is True for call in calls)
    assert all("symmetric_cmap" not in call for call in calls)
    assert (tmp_path / "thresholded.png").exists()
    assert (tmp_path / "unthresholded.png").exists()
