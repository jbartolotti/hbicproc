import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pipeline.cli import _build_parser
from pipeline.processing.qc import service
from pipeline.processing.qc import mask_qc


def test_qc_report_is_registered_as_global_cli_stage() -> None:
    args = _build_parser().parse_args(["qc_report"])

    assert args.command == "qc_report"
    assert not hasattr(args, "subject")


def test_motion_qc_generates_subject_mean_fd_report(tmp_path: Path, monkeypatch) -> None:
    confounds_one = tmp_path / "derivatives" / "fmriprep" / "sub-002" / "ses-baseline" / "func" / "sub-002_ses-baseline_task-rest_run-1_desc-confounds_timeseries.tsv"
    confounds_two = tmp_path / "derivatives" / "fmriprep" / "sub-001" / "ses-baseline" / "func" / "sub-001_ses-baseline_task-rest_run-1_desc-confounds_timeseries.tsv"
    confounds_one.parent.mkdir(parents=True)
    confounds_two.parent.mkdir(parents=True)
    confounds_one.write_text("framewise_displacement\n0.1\n0.3\n", encoding="utf-8")
    confounds_two.write_text("framewise_displacement\n0.2\nNaN\n0.4\n", encoding="utf-8")

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
            "reports": {
                "motion_qc": {
                    "enabled": True,
                    "fd_thresholds": {"warning": 0.2, "severe": 0.5},
                }
            },
        },
    }

    result = service.run_reports(config)

    assert result["success"] is True
    metadata = json.loads(
        (tmp_path / "qc_report" / "motion_qc" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["number_of_runs_analyzed"] == 2
    assert metadata["number_of_subjects_analyzed"] == 2
    assert metadata["summary"]["total runs"] == 2
    assert metadata["summary"]["overall mean FD"] == pytest.approx(0.2)
    assert metadata["summary"]["runs with >10% FD > 0.20"] == 2
    assert metadata["summary"]["runs with >10% FD > 0.50"] == 0
    assert (tmp_path / "qc_report" / "motion_qc" / "report.html").exists()
    assert (tmp_path / "qc_report" / "motion_qc" / "figures" / "mean_fd_histogram.png").exists()
    assert (tmp_path / "qc_report" / "motion_qc" / "figures" / "pct_fd_02_histogram.png").exists()
    assert (tmp_path / "qc_report" / "motion_qc" / "figures" / "pct_fd_05_histogram.png").exists()
    assert len(list((tmp_path / "qc_report" / "motion_qc" / "figures" / "fd_traces").glob("*.png"))) == 2
    assert len(list((tmp_path / "qc_report" / "motion_qc" / "figures" / "sorted_fd").glob("*.png"))) == 1
    report_html = (tmp_path / "qc_report" / "motion_qc" / "report.html").read_text(encoding="utf-8")
    assert "Ranked Runs" in report_html
    assert "Flagged Runs" in report_html
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


def test_contrast_motion_qc_discovers_contrasts_and_generates_maps(tmp_path: Path, monkeypatch) -> None:
    analysis_root = tmp_path / "derivatives" / "hbicproc"
    fmriprep_root = tmp_path / "derivatives" / "fmriprep"
    analysis_root.mkdir(parents=True)
    fmriprep_root.mkdir(parents=True)
    atlas = nib.Nifti1Image(np.array([[[1, 2, 1]]], dtype=np.int16), np.eye(4))

    confounds = []
    for index, subject in enumerate(("001", "002", "003"), start=1):
        confounds_path = fmriprep_root / f"sub-{subject}_task-nback_run-1_desc-confounds_timeseries.tsv"
        confounds_path.write_text(
            f"framewise_displacement\n{0.05 * index}\n{0.1 * index}\n", encoding="utf-8"
        )
        confounds.append(confounds_path)
        contrast_path = analysis_root / f"sub-{subject}_task-nback_run-1_desc-atlas-test-contrasts.tsv"
        contrast_path.write_text(
            "parcel_id\tparcel_label\tnetwork\themisphere\tcontrast\teffect\n"
            f"1\tParcel1\tDefault\tLH\tcondition_a\t{index}.0\n"
            f"2\tParcel2\tVisual\tRH\tcondition_a\t{4 - index}.0\n",
            encoding="utf-8",
        )
        nib.save(atlas, contrast_path.with_name(contrast_path.name.replace("-contrasts.tsv", "-resampled.nii.gz")))

    class FakeDatasetIndex:
        @classmethod
        def from_config(cls, config):
            return cls()

        def get_confounds_files(self, *, input_dataset):
            return confounds

    monkeypatch.setattr(service, "DatasetIndex", FakeDatasetIndex)
    config = {
        "log_dir": str(tmp_path / "logs"),
        "analysis": {
            "output_dir": str(analysis_root),
            "input_dataset": {"name": "fmriprep", "path": str(fmriprep_root)},
        },
        "qc_report": {
            "enabled": True,
            "output_dir": str(tmp_path / "qc_report"),
            "reports": {
                "contrast_motion_qc": {
                    "enabled": True,
                    "atlases": ["test"],
                    "motion_metrics": ["mean_fd"],
                }
            },
        },
    }

    result = service.run_reports(config)

    assert result["success"] is True
    report_root = tmp_path / "qc_report" / "contrast_motion_qc"
    metadata = json.loads((report_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["atlases"] == ["test"]
    assert metadata["contrasts"] == ["condition_a"]
    assert metadata["subject_count"] == 3
    assert metadata["run_count"] == 3
    assert list((report_root / "maps").glob("*.nii.gz"))
    assert list((report_root / "viewers").glob("*.html"))
    assert (report_root / "report.html").exists()


def test_mask_qc_groups_masks_and_writes_coverage_report(tmp_path: Path, monkeypatch) -> None:
    analysis_root = tmp_path / "derivatives" / "hbicproc"
    masks = [
        ("001", np.array([[[1, 1, 0, 0]]], dtype=np.uint8)),
        ("002", np.array([[[1, 1, 1, 0]]], dtype=np.uint8)),
        ("003", np.array([[[1, 0, 1, 1]]], dtype=np.uint8)),
    ]
    for subject, data in masks:
        path = (
            analysis_root
            / f"sub-{subject}"
            / "ses-BL"
            / "func"
            / "task-nback"
            / "run-1"
            / "models"
            / "glm"
            / f"sub-{subject}_ses-BL_task-nback_run-1_desc-mask.nii.gz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data, np.diag([2, 2, 2, 1])), path)

    class FakeViewer:
        def save_as_html(self, path):
            Path(path).write_text("<html>viewer</html>", encoding="utf-8")

    monkeypatch.setattr(mask_qc, "view_img", lambda *args, **kwargs: FakeViewer())
    monkeypatch.setattr(mask_qc, "_montage", lambda record, mask, reference, output_path, slices: output_path.write_bytes(b"png"))
    config = {
        "log_dir": str(tmp_path / "logs"),
        "analysis": {
            "output_dir": str(analysis_root),
            "input_dataset": {"name": "fmriprep", "path": str(tmp_path / "fmriprep")},
        },
        "qc_report": {
            "enabled": True,
            "output_dir": str(tmp_path / "qc_report"),
            "reports": {"mask_qc": {"enabled": True, "rare_voxel_threshold_pct": 50, "montage_slices": 6}},
        },
    }

    result = service.run_reports(config)

    assert result["success"] is True
    report_root = tmp_path / "qc_report" / "mask_qc"
    combo_root = report_root / "ses-BL" / "task-nback"
    metadata = json.loads((combo_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["number_of_masks_analyzed"] == 3
    assert metadata["rare_voxel_threshold_pct"] == 50
    coverage = nib.load(str(combo_root / "ses-BL_task-nback_mask_coverage.nii.gz")).get_fdata()
    np.testing.assert_array_equal(coverage, np.array([[[3, 2, 2, 1]]]))
    ranking = (combo_root / "report.html").read_text(encoding="utf-8")
    assert "Rare Voxel Ranking Table" in ranking
    assert "Subject Review Cards" in ranking
    assert (combo_root / "ses-BL_task-nback_coverage_viewer.html").exists()
    assert len(list((combo_root / "montages").glob("*.png"))) == 3
