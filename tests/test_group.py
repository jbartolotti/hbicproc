from pathlib import Path

import pandas as pd

from pipeline.processing.group.activation import run_activation
from pipeline.processing.group.atlas_metadata import AtlasMetadata
from pipeline.processing.group.discovery import discover_effect_maps
from pipeline.processing.group.participants import add_factor_columns, load_participants


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
