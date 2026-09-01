import json
import logging
from pathlib import Path

import pytest

from pipeline.cli import _build_parser
from pipeline.config import load_default_config
from pipeline.processing.analysis import AnalysisPluginRegistry, DatasetIndex
from pipeline.processing.analysis.activation.plugin import ActivationAnalysisPlugin
from pipeline.processing.analysis.derivatives import DerivativePathBuilder
from pipeline.stages import STAGE_CLASSES


def test_analysis_registry_registers_activation_plugin() -> None:
    registry = AnalysisPluginRegistry()
    registry.register(ActivationAnalysisPlugin)

    assert registry.get("activation") is ActivationAnalysisPlugin


def test_analysis_section_is_present_in_default_config() -> None:
    config = load_default_config(root_dir=".")

    assert "analysis" in config
    assert config["analysis"]["subject"]["activation"]["enabled"] is False


def test_analysis_files_are_in_processing_and_cli_command_exists() -> None:
    assert "analysis" in STAGE_CLASSES
    assert not (Path(__file__).resolve().parents[1] / "pipeline" / "analysis").exists()

    parser = _build_parser()
    args = parser.parse_args(["analysis", "sub-001"])

    assert args.command == "analysis"
    assert args.subject == "sub-001"


def test_activation_plugin_reads_repetition_time_from_bids_sidecar(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-rest_run-1_bold.nii.gz"
    bold_path.parent.mkdir(parents=True, exist_ok=True)
    bold_path.touch()
    sidecar = bold_path.with_name("sub-01_ses-01_task-rest_run-1_bold.json")
    sidecar.write_text(json.dumps({"RepetitionTime": 2.5}), encoding="utf-8")

    plugin = ActivationAnalysisPlugin()

    assert plugin._read_repetition_time(bold_path) == 2.5


def test_activation_plugin_requires_explicit_contrasts() -> None:
    plugin = ActivationAnalysisPlugin()
    result = plugin.run(
        "sub-01",
        {"study_root": "."},
        plugin_config={"enabled": True, "tasks": ["rest"], "contrasts": {}},
    )

    assert result.success is False
    assert "explicit" in result.message.lower()
    assert "contrasts" in result.message.lower()


def test_activation_plugin_expands_confounds_from_config() -> None:
    plugin = ActivationAnalysisPlugin()
    expanded = plugin._expand_confounds(
        {
            "motion": True,
            "motion_derivatives": True,
            "framewise_displacement": True,
            "acompcor": 5,
        }
    )

    assert "trans_x" in expanded
    assert "trans_y_derivative1" in expanded
    assert "framewise_displacement" in expanded
    assert "a_comp_cor_04" in expanded


def test_dataset_index_normalizes_session_and_run_entities_without_prefix() -> None:
    dataset = DatasetIndex(".")
    entities = {"subject": "01", "session": "baseline", "task": "rest", "run": "1"}

    assert dataset._entity_value(entities, "session") == "baseline"
    assert "ses-" not in dataset._entity_value(entities, "session")
    assert dataset._entity_value(entities, "run") == "1"


def test_derivative_path_builder_replaces_underscores_with_hyphens() -> None:
    path = DerivativePathBuilder.build(
        "derivatives",
        subject="sub-01",
        session="baseline_session",
        task="rest_task",
        run="run-2",
        desc="condition_a",
        stat="effect_map",
    )

    assert path.name == "sub-01_ses-baseline-session_task-rest-task_run-2_desc-condition-a_stat-effect-map.nii.gz"


def test_derivative_path_builder_builds_session_directory_with_bids_prefix() -> None:
    directory = DerivativePathBuilder.build_directory("derivatives/hbicproc", subject="sub-01", session="baseline")

    assert directory == Path("derivatives/hbicproc/sub-01/ses-baseline")


def test_dataset_index_logs_discovery(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    bids_root.mkdir(parents=True, exist_ok=True)
    (bids_root / "dataset_description.json").write_text('{"Name": "Test dataset", "BIDSVersion": "1.8.0"}', encoding="utf-8")

    raw_func_dir = bids_root / "sub-001" / "ses-baseline" / "func"
    derivative_func_dir = bids_root / "derivatives" / "fmriprep" / "sub-001" / "ses-baseline" / "func"
    raw_func_dir.mkdir(parents=True, exist_ok=True)
    derivative_func_dir.mkdir(parents=True, exist_ok=True)
    (bids_root / "derivatives" / "fmriprep" / "dataset_description.json").write_text(
        '{"Name": "fMRIPrep", "BIDSVersion": "1.8.0"}',
        encoding="utf-8",
    )
    (derivative_func_dir / "sub-001_ses-baseline_task-rest_run-1_desc-preproc_bold.nii.gz").touch()
    (raw_func_dir / "sub-001_ses-baseline_task-rest_run-1_events.tsv").touch()
    (derivative_func_dir / "sub-001_ses-baseline_task-rest_run-1_desc-confounds_timeseries.tsv").touch()

    with caplog.at_level(logging.INFO):
        dataset = DatasetIndex(bids_root)
        runs = dataset.get_task_runs(subject="001", task="rest")

    assert len(runs) == 1
    assert "Dataset discovery requested" in caplog.text
    assert "Discovered BOLD files" in caplog.text
    assert "Resolved AnalysisRun objects" in caplog.text


def test_dataset_index_matches_exact_bids_entities(tmp_path: Path) -> None:
    study_root = tmp_path
    raw_func_dir = study_root / "sub-01" / "func"
    derivative_root = study_root / "derivatives" / "custom-preproc"
    raw_func_dir.mkdir(parents=True, exist_ok=True)
    derivative_root.mkdir(parents=True, exist_ok=True)
    (derivative_root / "dataset_description.json").write_text(
        '{"Name": "Custom preprocessing", "BIDSVersion": "1.8.0"}',
        encoding="utf-8",
    )

    (raw_func_dir / "sub-01_task-rest_run-1_events.tsv").touch()
    (raw_func_dir / "sub-010_task-rest_run-2_events.tsv").touch()

    for subject, run in (("01", "1"), ("010", "2")):
        derivative_func_dir = derivative_root / f"sub-{subject}" / "func"
        derivative_func_dir.mkdir(parents=True, exist_ok=True)
        (derivative_func_dir / f"sub-{subject}_task-rest_run-{run}_desc-preproc_bold.nii.gz").touch()
        (derivative_func_dir / f"sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv").touch()

    config = {"study_root": str(study_root), "bids_root": str(study_root), "analysis": {"derivative_dataset": "custom-preproc"}}

    dataset = DatasetIndex.from_config(config)
    run_infos = [run.as_dict() for run in dataset.get_task_runs(subject="01", task="rest")]

    assert len(run_infos) == 1
    assert run_infos[0]["run"] == "1"
    assert Path(run_infos[0]["bold_path"]).name.startswith("sub-01_task-rest_run-1")

    plugin = ActivationAnalysisPlugin()
    with pytest.raises(ValueError, match="RepetitionTime"):
        missing = study_root / "sub-02" / "func" / "sub-02_task-rest_run-1_bold.nii.gz"
        missing.parent.mkdir(parents=True, exist_ok=True)
        missing.touch()
        plugin._read_repetition_time(missing)
