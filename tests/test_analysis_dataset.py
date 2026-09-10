from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.processing.analysis import InputDataset, TaskRunContext
from pipeline.processing.analysis.dataset import DatasetIndex, regenerate_cached_indexes
from pipeline.state import invalidate_bids_index
from pipeline.processing.analysis.derivatives import DerivativePathBuilder
from pipeline.cli import _build_parser


def test_index_cli_accepts_dataset_selector() -> None:
    args = _build_parser().parse_args(["index", "--dataset", "raw"])

    assert args.dataset == "raw"


def test_regenerate_cached_indexes_can_select_one_dataset(tmp_path: Path, monkeypatch) -> None:
    config = {
        "bids_root": str(tmp_path),
        "analysis": {
            "input_dataset": {
                "name": "fmriprep",
                "path": str(tmp_path / "derivatives" / "fmriprep"),
            }
        },
    }
    invalidated = []
    constructed = []

    class FakeDatasetIndex:
        @classmethod
        def from_config(cls, config, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr("pipeline.processing.analysis.dataset.DatasetIndex", FakeDatasetIndex)
    monkeypatch.setattr(
        "pipeline.processing.analysis.dataset.invalidate_bids_index",
        lambda name, root: invalidated.append((name, root)),
    )

    regenerate_cached_indexes(config, dataset_name="raw")
    assert invalidated == [("raw", tmp_path)]
    assert constructed == [{"build_raw_index": True, "build_derivative_index": False}]

    invalidated.clear()
    constructed.clear()
    regenerate_cached_indexes(config, dataset_name="fmriprep")
    assert invalidated == [("fmriprep", tmp_path)]
    assert constructed == [{
        "task_config": {"input_dataset": config["analysis"]["input_dataset"]},
        "build_raw_index": False,
        "build_derivative_index": True,
    }]


def test_regenerate_cached_indexes_rejects_unknown_dataset(tmp_path: Path) -> None:
    config = {
        "bids_root": str(tmp_path),
        "analysis": {
            "input_dataset": {"name": "fmriprep", "path": str(tmp_path / "fmriprep")}
        },
    }

    try:
        regenerate_cached_indexes(config, dataset_name="missing")
    except ValueError as exc:
        assert "Unknown dataset 'missing'" in str(exc)
    else:
        raise AssertionError("Expected an unknown dataset selector to fail.")


def test_regenerate_cached_indexes_defaults_to_all(tmp_path: Path, monkeypatch) -> None:
    config = {
        "bids_root": str(tmp_path),
        "analysis": {
            "input_dataset": {"name": "fmriprep", "path": str(tmp_path / "fmriprep")}
        },
    }
    invalidated = []
    constructed = []

    class FakeDatasetIndex:
        @classmethod
        def from_config(cls, config, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setattr("pipeline.processing.analysis.dataset.DatasetIndex", FakeDatasetIndex)
    monkeypatch.setattr(
        "pipeline.processing.analysis.dataset.invalidate_bids_index",
        lambda name, root: invalidated.append(name),
    )

    regenerate_cached_indexes(config)

    assert invalidated == ["raw", "fmriprep"]
    assert constructed == [{}, {"task_config": {"input_dataset": config["analysis"]["input_dataset"]}}]


def test_dataset_index_resolves_runs_from_configured_input_dataset(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    derivative_root = bids_root / "derivatives" / "fmriprep"
    raw_func_dir = bids_root / "sub-001" / "ses-baseline" / "func"
    derivative_func_dir = derivative_root / "sub-001" / "ses-baseline" / "func"
    raw_func_dir.mkdir(parents=True, exist_ok=True)
    derivative_func_dir.mkdir(parents=True, exist_ok=True)

    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test dataset", "BIDSVersion": "1.8.0"}', encoding="utf-8"
    )
    (derivative_root / "dataset_description.json").write_text(
        '{"Name": "fMRIPrep", "BIDSVersion": "1.8.0"}', encoding="utf-8"
    )

    bold_path = derivative_func_dir / "sub-001_ses-baseline_task-rest_run-1_desc-preproc_bold.nii.gz"
    events_path = raw_func_dir / "sub-001_ses-baseline_task-rest_run-1_events.tsv"
    confounds_path = derivative_func_dir / "sub-001_ses-baseline_task-rest_run-1_desc-confounds_timeseries.tsv"
    bold_path.write_bytes(b"fake-nifti")
    events_path.write_text("onset\tduration\ttrial_type\n0\t1\tcontrol\n", encoding="utf-8")
    confounds_path.write_text("trans_x\trot_y\n0.1\t0.2\n", encoding="utf-8")

    dataset = DatasetIndex.from_config(
        {
            "bids_root": str(bids_root),
            "study_root": str(tmp_path),
            "analysis": {
                "input_dataset": {
                    "name": "fmriprep",
                    "path": str(derivative_root),
                }
            },
        }
    )
    runs = dataset.get_task_runs(subject="001", task="rest")

    assert len(runs) == 1
    assert runs[0] == TaskRunContext(
        subject="001",
        session="baseline",
        task="rest",
        run="1",
        bold_path=bold_path,
        events_path=events_path,
        confounds_path=confounds_path,
        input_dataset=InputDataset("fmriprep", derivative_root),
    )


def test_dataset_index_supports_task_specific_input_dataset(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    derivative_root = bids_root / "derivatives" / "afni"
    raw_func_dir = bids_root / "sub-008" / "func"
    derivative_func_dir = derivative_root / "sub-008" / "func"
    raw_func_dir.mkdir(parents=True, exist_ok=True)
    derivative_func_dir.mkdir(parents=True, exist_ok=True)
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test dataset", "BIDSVersion": "1.8.0"}', encoding="utf-8"
    )
    (derivative_root / "dataset_description.json").write_text(
        '{"Name": "AFNI", "BIDSVersion": "1.8.0"}', encoding="utf-8"
    )
    (raw_func_dir / "sub-008_task-nback_events.tsv").touch()
    (derivative_func_dir / "sub-008_task-nback_desc-preproc_bold.nii.gz").touch()
    (derivative_func_dir / "sub-008_task-nback_desc-confounds_timeseries.tsv").touch()

    dataset = DatasetIndex.from_config(
        {
            "bids_root": str(bids_root),
            "analysis": {
                "input_dataset": {
                    "name": "fmriprep",
                    "path": str(bids_root / "derivatives" / "fmriprep"),
                }
            },
            "study_root": str(tmp_path),
        },
        task_config={"input_dataset": {"name": "afni", "path": str(derivative_root)}},
    )
    runs = dataset.get_task_runs(
        subject="sub-008",
        task="task-nback",
        input_dataset=InputDataset("afni", derivative_root),
    )

    assert [(run.subject, run.task, run.input_dataset.name) for run in runs] == [("008", "nback", "afni")]


def test_pybids_index_cache_creation_reuse_and_invalidation(
    tmp_path: Path,
    caplog,
) -> None:
    bids_root = tmp_path / "bids"
    derivative_root = bids_root / "derivatives" / "fmriprep"
    bids_root.mkdir(parents=True)
    derivative_root.mkdir(parents=True)
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test dataset", "BIDSVersion": "1.8.0"}', encoding="utf-8"
    )
    (derivative_root / "dataset_description.json").write_text(
        '{"Name": "fMRIPrep", "BIDSVersion": "1.8.0"}', encoding="utf-8"
    )
    config = {
        "bids_root": str(bids_root),
        "study_root": str(tmp_path),
        "analysis": {
            "input_dataset": {"name": "fmriprep", "path": str(derivative_root)},
        },
    }

    caplog.set_level(logging.INFO)
    DatasetIndex.from_config(config)
    cache_dir = bids_root / "code" / "cache" / "pybids"
    state_path = bids_root / "code" / "cache" / "pipeline_state.json"
    assert cache_dir.exists()
    assert state_path.exists()
    assert {path.name.split("-", 1)[0] for path in cache_dir.glob("*.json")} == {"raw", "fmriprep"}
    assert json.loads(state_path.read_text(encoding="utf-8"))["bids_indexes"] == {
        "raw": 0,
        "fmriprep": 0,
    }

    caplog.clear()
    DatasetIndex.from_config(config)
    assert "Using cached PyBIDS index raw" in caplog.text
    assert "Using cached PyBIDS index fmriprep" in caplog.text

    assert invalidate_bids_index("fmriprep", bids_root) == 1
    caplog.clear()
    DatasetIndex.from_config(config)
    assert "PyBIDS cache stale, rebuilding fmriprep" in caplog.text
    assert "Using cached PyBIDS index raw" in caplog.text
    metadata = next(cache_dir.glob("fmriprep-*.json"))
    assert json.loads(metadata.read_text(encoding="utf-8"))["revision"] == 1


def test_derivative_path_builder_separates_model_and_analysis_namespaces(tmp_path: Path) -> None:
    context = TaskRunContext(subject="sub-01", session="baseline", task="rest", run="2")
    model = DerivativePathBuilder.for_context(tmp_path, context, namespace="model", name="canonical_glm")
    analysis = DerivativePathBuilder.for_context(tmp_path, context, namespace="analysis", name="canonical_glm")

    expected_base = tmp_path / "sub-01" / "ses-baseline" / "func" / "task-rest" / "run-2"
    assert model.model_directory == expected_base / "models" / "canonical_glm"
    assert analysis.analysis_directory == expected_base / "analyses" / "canonical_glm"
    assert model.model_directory != analysis.analysis_directory
