import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from pipeline.cli import _build_parser
from pipeline.config import load_default_config
from pipeline.processing.analysis import AnalysisPluginRegistry, AnalysisRun, DatasetIndex
from pipeline.processing.analysis.activation.plugin import ActivationAnalysisPlugin
from pipeline.processing.analysis.derivatives import DerivativePathBuilder
from pipeline.processing.analysis import service as analysis_service
from pipeline.stages import STAGE_CLASSES
from pipeline.stages.analysis import AnalysisStage


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


def test_analysis_stage_ignores_subject_level_completion_state(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_analysis_run(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        _ = (args, kwargs)
        called = True
        return {"success": True, "skipped": True, "message": "run invoked", "details": {}}

    monkeypatch.setattr("pipeline.stages.analysis.analysis_run", fake_analysis_run)

    stage = AnalysisStage()
    result = stage.execute("001", {}, {"analysis_complete": True})

    assert stage.state_key is None
    assert called is True
    assert result.skipped is True


def test_analysis_stage_reuses_dataset_index_across_subjects(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []
    received: list[object] = []

    class FakeDatasetIndex:
        pass

    def fake_from_config(config: dict[str, object]) -> FakeDatasetIndex:
        _ = config
        index = FakeDatasetIndex()
        created.append(index)
        return index

    def fake_analysis_run(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args
        received.append(kwargs["dataset_index"])
        return {"success": True, "skipped": True, "message": "run invoked", "details": {}}

    monkeypatch.setattr("pipeline.stages.analysis.DatasetIndex.from_config", fake_from_config)
    monkeypatch.setattr("pipeline.stages.analysis.analysis_run", fake_analysis_run)

    config = {"analysis": {"subject": {"activation": {"enabled": True}}}}
    stage = AnalysisStage()
    with caplog.at_level(logging.INFO):
        stage.prepare(config)
        stage.execute("001", config, {})
        stage.execute("002", config, {})

    assert len(created) == 1
    assert received == [created[0], created[0]]
    assert "BIDS indexing" in caplog.text


def test_analysis_plugin_load_failure_is_logged_and_raised(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = AnalysisPluginRegistry()

    def fail_register(plugin_cls: type) -> type:
        _ = plugin_cls
        raise ImportError("analysis outputs module is unavailable")

    monkeypatch.setattr(registry, "register", fail_register)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Failed to load built-in analysis plugin"):
            registry._load_builtin_plugins()

    assert "analysis outputs module is unavailable" in caplog.text


def test_analysis_service_reports_registry_loading_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_discovery() -> AnalysisPluginRegistry:
        raise RuntimeError("missing analysis outputs module")

    monkeypatch.setattr(analysis_service, "get_analysis_registry", fail_discovery)

    result = analysis_service.run("001", {})

    assert result["success"] is False
    assert result["skipped"] is False
    assert "plugin loading failed" in result["message"].lower()
    assert "missing analysis outputs module" in result["message"]


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


def test_activation_plugin_saves_design_png_from_nilearn_axes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeFigure:
        def savefig(self, path: Path, **kwargs: object) -> None:
            _ = kwargs
            Path(path).write_bytes(b"png")

    class FakeAxes:
        figure = FakeFigure()

    monkeypatch.setattr(
        "nilearn.plotting.plot_design_matrix",
        lambda design_matrix, rescale: (design_matrix, rescale, FakeAxes())[-1],
    )

    output_path = tmp_path / "design_matrix.png"
    ActivationAnalysisPlugin()._save_design_png(pd.DataFrame({"condition": [0.0, 1.0]}), output_path)

    assert output_path.read_bytes() == b"png"


def test_activation_png_failure_does_not_prevent_scientific_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_order: list[str] = []

    class FakeImage:
        def __init__(self, label: str) -> None:
            self.label = label

        def to_filename(self, path: Path) -> None:
            output_order.append(self.label)
            Path(path).write_bytes(self.label.encode("utf-8"))

    class FakeReport:
        def save_as_html(self, path: Path) -> None:
            output_order.append("report")
            Path(path).write_text("<html></html>", encoding="utf-8")

    class FakeMasker:
        mask_img_ = FakeImage("mask")

    class FakeFirstLevelModel:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs
            self.design_matrices_: list[pd.DataFrame] = []
            self.residuals_ = [FakeImage("residual")]
            self.masker_ = FakeMasker()

        def fit(self, _bold_path: Path, *, events: pd.DataFrame, confounds: None) -> None:
            _ = (_bold_path, events, confounds)
            self.design_matrices_ = [pd.DataFrame({"condition": [0.0, 1.0]})]

        def compute_contrast(self, _expression: str, *, output_type: str) -> FakeImage:
            _ = _expression
            return FakeImage(output_type)

        def generate_report(self, *, contrasts: dict[str, str]) -> FakeReport:
            _ = contrasts
            return FakeReport()

    monkeypatch.setattr("nilearn.glm.first_level.FirstLevelModel", FakeFirstLevelModel)

    plugin = ActivationAnalysisPlugin()

    def fake_read_repetition_time(bold_path: Path) -> float:
        _ = bold_path
        return 2.0

    monkeypatch.setattr(
        plugin,
        "_read_repetition_time",
        fake_read_repetition_time,
    )

    def fail_design_png(_design_matrix: pd.DataFrame, _path: Path) -> None:
        _ = (_design_matrix, _path)
        output_order.append("design_png")
        raise RuntimeError("optional PNG failure")

    monkeypatch.setattr(plugin, "_save_design_png", fail_design_png)

    events_path = tmp_path / "events.tsv"
    events_path.write_text("onset\tduration\ttrial_type\n0\t1\tcondition\n", encoding="utf-8")
    confounds_path = tmp_path / "confounds.tsv"
    confounds_path.write_text("trans_x\n0.1\n", encoding="utf-8")
    bold_path = tmp_path / "bold.nii.gz"
    bold_path.touch()

    result = plugin._run_single_run(
        "001",
        "rest",
        AnalysisRun(
            subject="001",
            session="BL",
            task="rest",
            run="1",
            bold_path=bold_path,
            events_path=events_path,
            confounds_path=confounds_path,
        ),
        {"study_root": str(tmp_path)},
        {"enabled": True, "contrasts": {"condition": "condition"}},
    )

    assert result.success is True
    assert output_order.index("design_png") > output_order.index("mask")
    assert output_order[-1] == "report"
    assert len(list((tmp_path / "derivatives" / "hbicproc" / "sub-001" / "ses-BL" / "func").glob("*.nii.gz"))) >= 5

    skipped_result = plugin._run_single_run(
        "001",
        "rest",
        AnalysisRun(
            subject="001",
            session="BL",
            task="rest",
            run="1",
            bold_path=bold_path,
            events_path=events_path,
            confounds_path=confounds_path,
        ),
        {"study_root": str(tmp_path)},
        {"enabled": True, "contrasts": {"condition": "condition"}},
    )

    assert skipped_result.success is True
    assert skipped_result.skipped is True
    assert skipped_result.details["output_based_skip"] is True

    rerun_result = plugin._run_single_run(
        "001",
        "rest",
        AnalysisRun(
            subject="001",
            session="BL",
            task="rest",
            run="1",
            bold_path=bold_path,
            events_path=events_path,
            confounds_path=confounds_path,
        ),
        {"study_root": str(tmp_path)},
        {"enabled": True, "contrasts": {"condition": "condition"}},
        rerun=True,
    )

    assert rerun_result.success is True
    assert rerun_result.skipped is False


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
        repeated_runs = dataset.get_task_runs(subject="sub-001", task="task-rest")

    assert len(runs) == 1
    assert repeated_runs == runs
    assert "Reusing cached dataset discovery" in caplog.text
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
