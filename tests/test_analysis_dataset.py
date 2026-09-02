from __future__ import annotations

from pathlib import Path

from pipeline.processing.analysis import AnalysisOutputInventory, AnalysisRun
from pipeline.processing.analysis.dataset import DatasetIndex


def test_dataset_index_resolves_bids_runs(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    bids_root.mkdir(parents=True, exist_ok=True)
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test dataset", "BIDSVersion": "1.8.0"}',
        encoding="utf-8",
    )

    raw_func_dir = bids_root / "sub-001" / "ses-baseline" / "func"
    derivative_func_dir = bids_root / "derivatives" / "fmriprep" / "sub-001" / "ses-baseline" / "func"
    raw_func_dir.mkdir(parents=True, exist_ok=True)
    derivative_func_dir.mkdir(parents=True, exist_ok=True)

    bold_path = derivative_func_dir / "sub-001_ses-baseline_task-rest_run-1_desc-preproc_bold.nii.gz"
    events_path = raw_func_dir / "sub-001_ses-baseline_task-rest_run-1_events.tsv"
    confounds_path = derivative_func_dir / "sub-001_ses-baseline_task-rest_run-1_desc-confounds_timeseries.tsv"

    (bids_root / "derivatives" / "fmriprep" / "dataset_description.json").write_text(
        '{"Name": "fMRIPrep", "BIDSVersion": "1.8.0"}',
        encoding="utf-8",
    )
    bold_path.write_bytes(b"fake-nifti")
    events_path.write_text("onset\tduration\ttrial_type\n0\t1\tcontrol\n", encoding="utf-8")
    confounds_path.write_text("trans_x\trot_y\n0.1\t0.2\n", encoding="utf-8")

    dataset = DatasetIndex(bids_root)
    runs = dataset.get_task_runs(subject="001", task="rest")

    assert len(runs) == 1
    run = runs[0]
    assert run.subject == "001"
    assert run.session == "baseline"
    assert run.run == "1"
    assert run.task == "rest"
    assert run.bold_path == bold_path
    assert run.events_path == events_path
    assert run.confounds_path == confounds_path


def test_dataset_index_resolves_all_sessions_from_configured_derivative(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    derivative_root = bids_root / "derivatives" / "afni"
    bids_root.mkdir(parents=True, exist_ok=True)
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test dataset", "BIDSVersion": "1.8.0"}',
        encoding="utf-8",
    )
    derivative_root.mkdir(parents=True, exist_ok=True)
    (derivative_root / "dataset_description.json").write_text(
        '{"Name": "AFNI", "BIDSVersion": "1.8.0"}',
        encoding="utf-8",
    )

    for session in ("BL", "w12"):
        raw_func = bids_root / "sub-008" / f"ses-{session}" / "func"
        derivative_func = derivative_root / "sub-008" / f"ses-{session}" / "func"
        raw_func.mkdir(parents=True, exist_ok=True)
        derivative_func.mkdir(parents=True, exist_ok=True)
        (raw_func / f"sub-008_ses-{session}_task-nback_events.tsv").touch()
        (derivative_func / f"sub-008_ses-{session}_task-nback_desc-preproc_bold.nii.gz").touch()
        (derivative_func / f"sub-008_ses-{session}_task-nback_desc-confounds_timeseries.tsv").touch()

    dataset = DatasetIndex.from_config(
        {
            "bids_root": str(bids_root),
            "analysis": {"derivative_dataset": "afni"},
        }
    )
    runs = dataset.get_task_runs(subject="sub-008", task="task-nback")

    assert [(run.subject, run.session, run.task) for run in runs] == [
        ("008", "BL", "nback"),
        ("008", "w12", "nback"),
    ]


def test_analysis_output_inventory_requires_all_configured_contrasts(tmp_path: Path) -> None:
    analysis_run = AnalysisRun(subject="008", session="BL", task="nback", run="1")
    inventory = AnalysisOutputInventory.for_run(
        tmp_path / "func",
        analysis_run,
        ("2back_gt_baseline", "1back_gt_baseline", "2back_gt_1back"),
    )

    required_outputs = inventory.required_outputs()
    assert len(required_outputs) == 10
    assert len(inventory.missing_outputs()) == 10

    for path in required_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert inventory.is_complete() is True
