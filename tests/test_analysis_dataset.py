from __future__ import annotations

from pathlib import Path

from pipeline.processing.analysis.dataset import DatasetIndex


def test_dataset_index_resolves_bids_runs(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    bids_root.mkdir(parents=True, exist_ok=True)
    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test dataset", "BIDSVersion": "1.8.0"}',
        encoding="utf-8",
    )

    func_dir = bids_root / "sub-001" / "ses-baseline" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    bold_path = func_dir / "sub-001_ses-baseline_task-rest_run-1_bold.nii.gz"
    events_path = func_dir / "sub-001_ses-baseline_task-rest_run-1_events.tsv"
    confounds_path = func_dir / "sub-001_ses-baseline_task-rest_run-1_desc-confounds_timeseries.tsv"

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
