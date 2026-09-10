import json
from pathlib import Path

import pytest

from pipeline.cli import _build_parser
from pipeline.processing.qc import service


def test_qc_report_is_registered_as_global_cli_stage() -> None:
    args = _build_parser().parse_args(["qc_report"])

    assert args.command == "qc_report"
    assert not hasattr(args, "subject")


def test_motion_qc_generates_subject_mean_fd_report(tmp_path: Path, monkeypatch) -> None:
    confounds_one = tmp_path / "derivatives" / "fmriprep" / "sub-002" / "func" / "sub-002_task-rest_desc-confounds_timeseries.tsv"
    confounds_two = tmp_path / "derivatives" / "fmriprep" / "sub-001" / "func" / "sub-001_task-rest_desc-confounds_timeseries.tsv"
    confounds_one.parent.mkdir(parents=True)
    confounds_two.parent.mkdir(parents=True)
    confounds_one.write_text("framewise_displacement\n0.1\n0.3\n", encoding="utf-8")
    confounds_two.write_text("framewise_displacement\n0.2\n0.4\n", encoding="utf-8")

    class FakeDatasetIndex:
        @classmethod
        def from_config(cls, config):
            return cls()

        def get_confounds_files(self, *, input_dataset):
            assert input_dataset.name == "fmriprep"
            return [confounds_one, confounds_two]

    monkeypatch.setattr(service, "DatasetIndex", FakeDatasetIndex)
    config = {
        "log_dir": str(tmp_path / "logs"),
        "analysis": {
            "input_dataset": {
                "name": "fmriprep",
                "path": str(tmp_path / "derivatives" / "fmriprep"),
            }
        },
        "qc_report": {
            "enabled": True,
            "output_dir": str(tmp_path / "qc_report"),
            "reports": {"motion_qc": {"enabled": True}},
        },
    }

    result = service.run_reports(config)

    assert result["success"] is True
    metadata = json.loads(
        (tmp_path / "qc_report" / "motion_qc" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["mean_framewise_displacement"] == pytest.approx(
        {"sub-001": 0.3, "sub-002": 0.2}
    )
    assert (tmp_path / "qc_report" / "motion_qc" / "report.html").exists()
    assert (tmp_path / "qc_report" / "motion_qc" / "figures" / "mean_framewise_displacement.png").exists()
    assert (tmp_path / "qc_report" / "index.html").exists()
    assert (tmp_path / "qc_report" / "manifest.json").exists()


def test_motion_qc_skips_when_confounds_are_unavailable(tmp_path: Path, monkeypatch) -> None:
    class FakeDatasetIndex:
        @classmethod
        def from_config(cls, config):
            return cls()

        def get_confounds_files(self, *, input_dataset):
            return []

    monkeypatch.setattr(service, "DatasetIndex", FakeDatasetIndex)
    config = {
        "log_dir": str(tmp_path / "logs"),
        "analysis": {
            "input_dataset": {"name": "fmriprep", "path": str(tmp_path / "fmriprep")}
        },
        "qc_report": {
            "enabled": True,
            "output_dir": str(tmp_path / "qc_report"),
            "reports": {"motion_qc": {"enabled": True}},
        },
    }

    result = service.run_reports(config)

    assert result["success"] is True
    assert result["details"]["reports"][0]["status"] == "skipped"
