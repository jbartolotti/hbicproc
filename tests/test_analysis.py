import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from pipeline.cli import _build_parser
from pipeline.config import load_config, load_default_config
from pipeline.processing.analysis import (
    AnalysisSpec,
    InputDataset,
    ModelSpec,
    TaskRunContext,
    build_task_plans,
)
from pipeline.processing.analysis.analyses.activation import ActivationAnalysis
from pipeline.processing.analysis.analyses.atlas import AtlasCache
from pipeline.processing.analysis.models.canonical_glm import FittedModel
from pipeline.processing.analysis.derivatives import DerivativePathBuilder
from pipeline.processing.analysis.models.canonical_glm import CanonicalGLMModel
from pipeline.processing.analysis.service import run
from pipeline.stages import STAGE_CLASSES


def test_analysis_section_uses_task_centered_defaults() -> None:
    config = load_default_config(root_dir=".")

    assert config["analysis"]["output_dir"].endswith("derivatives\\hbicproc")
    assert config["analysis"]["input_dataset"]["name"] == "fmriprep"
    assert config["analysis"]["subject"]["tasks"] == {}


def test_configured_tasks_default_to_enabled_and_resolve_dataset_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(
        json.dumps(
            {
                "study_root": ".",
                "analysis": {
                    "input_dataset": {"name": "fmriprep", "path": "derivatives/fmriprep"},
                    "subject": {
                        "tasks": {
                            "nback": {
                                "models": {"canonical_glm": {"type": "canonical_glm"}},
                                "analyses": {"activation": {"model": "canonical_glm"}},
                            },
                            "rest": {
                                "enabled": False,
                                "input_dataset": {"name": "afni", "path": "derivatives/afni"},
                                "models": {"afni_glm": {"type": "afni"}},
                                "analyses": {"connectivity": {"model": "afni_glm"}},
                            },
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["analysis"]["subject"]["tasks"]["nback"]["enabled"] is True
    assert config["analysis"]["subject"]["tasks"]["rest"]["enabled"] is False
    assert Path(config["analysis"]["input_dataset"]["path"]).is_absolute()
    assert Path(config["analysis"]["subject"]["tasks"]["rest"]["input_dataset"]["path"]).is_absolute()


def test_old_activation_configuration_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(
        json.dumps({"analysis": {"subject": {"activation": {"enabled": True}}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="activation is obsolete"):
        load_config(config_path)


def test_analysis_command_names_are_preserved() -> None:
    assert {"analysis", "bidsify", "validate", "qc_review"}.issubset(STAGE_CLASSES)

    parser = _build_parser()
    args = parser.parse_args(["analysis", "sub-001"])

    assert args.command == "analysis"
    assert args.subject == "sub-001"
    assert not hasattr(args, "plugin")


def test_task_plans_bind_global_and_task_specific_input_datasets(tmp_path: Path) -> None:
    config = {
        "analysis": {
            "output_dir": str(tmp_path / "derivatives"),
            "input_dataset": {"name": "fmriprep", "path": str(tmp_path / "fmriprep")},
            "subject": {
                "tasks": {
                    "nback": {
                        "models": {
                            "canonical_glm": {
                                "type": "canonical_glm",
                                "atlases": ["schaefer200"],
                            }
                        },
                        "analyses": {
                            "activation": {
                                "model": "canonical_glm",
                                "requires": ["beta_maps"],
                            }
                        },
                    },
                    "rest": {
                        "input_dataset": {"name": "afni", "path": str(tmp_path / "afni")},
                        "models": {"afni_glm": {"type": "afni"}},
                        "analyses": {"connectivity": {"model": "afni_glm"}},
                    },
                }
            },
        }
    }

    plans = build_task_plans(config, "001")

    assert [plan.task for plan in plans] == ["nback", "rest"]
    assert plans[0].input_dataset == InputDataset("fmriprep", tmp_path / "fmriprep")
    assert plans[1].input_dataset == InputDataset("afni", tmp_path / "afni")
    assert plans[0].analyses[0].required_derivatives == ("beta_maps",)


def test_model_fingerprint_excludes_atlas_configuration() -> None:
    model = ModelSpec(
        name="canonical_glm",
        model_type="canonical_glm",
        configuration={"smoothing_fwhm": 6.0, "atlases": ["schaefer200"]},
    )
    context = TaskRunContext(subject="001", task="nback", run="1")

    assert "atlases" not in model.fingerprint_configuration
    assert model.fingerprint_payload(context)["run"] == "1"


def test_model_and_analysis_namespaces_are_separate_and_run_scoped(tmp_path: Path) -> None:
    context = TaskRunContext(subject="001", session="01", task="nback", run="1")
    model = ModelSpec("activation", "canonical_glm", {})
    analysis = AnalysisSpec("activation", "activation", {})

    model_path = model.plan(context, str(tmp_path)).derivatives.model_directory
    analysis_path = analysis.plan(context, model, str(tmp_path)).derivatives.analysis_directory

    assert model_path == tmp_path / "sub-001" / "ses-01" / "func" / "task-nback" / "run-1" / "models" / "activation"
    assert analysis_path == tmp_path / "sub-001" / "ses-01" / "func" / "task-nback" / "run-1" / "analyses" / "activation"
    assert model_path != analysis_path


def test_analysis_service_plans_without_inputs_and_does_not_fit(tmp_path: Path) -> None:
    config = {
        "analysis": {
            "output_dir": str(tmp_path / "derivatives"),
            "input_dataset": {"name": "fmriprep", "path": str(tmp_path / "fmriprep")},
            "subject": {
                "tasks": {
                    "nback": {
                        "models": {"canonical_glm": {"type": "canonical_glm"}},
                        "analyses": {"activation": {"model": "canonical_glm"}},
                    }
                }
            },
        }
    }

    result = run("001", config)

    assert result["success"] is True
    assert result["details"]["models"][0]["model_type"] == "canonical_glm"
    assert "fitted 0 model run(s)" in result["message"]
    assert result["details"]["fitted_models"] == []
    assert not (tmp_path / "derivatives").exists()


def test_derivative_path_builder_rejects_namespace_path_traversal() -> None:
    context = TaskRunContext(subject="001", task="nback")

    with pytest.raises(ValueError, match="single path components"):
        DerivativePathBuilder.for_context("derivatives", context, namespace="model", name="../model")


def test_canonical_glm_fits_one_run_with_events_and_confounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bold_path = tmp_path / "bold.nii.gz"
    events_path = tmp_path / "events.tsv"
    confounds_path = tmp_path / "confounds.tsv"
    bold_path.touch()
    events_path.write_text("onset\tduration\ttrial_type\n0\t1\tone\n", encoding="utf-8")
    confounds_path.write_text("trans_x\trot_y\n0.1\t0.2\n", encoding="utf-8")

    class FakeGLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.design_matrices_ = []

        def fit(self, image: str, *, events: object, confounds: object) -> "FakeGLM":
            self.image = image
            self.events = events
            self.confounds = confounds
            self.design_matrices_ = [pd.DataFrame([[1.0, 0.1]], columns=["one", "trans_x"])]
            result = type(
                "Result",
                (),
                {
                    "theta": np.array([[1.0], [2.0]]),
                    "cov": np.eye(2),
                    "dispersion": np.array([0.5]),
                    "df_residuals": 10,
                },
            )()
            self.results_ = [{"0": result}]
            return self

    monkeypatch.setattr("pipeline.processing.analysis.models.canonical_glm.FirstLevelModel", FakeGLM)
    spec = ModelSpec(
        "canonical_glm",
        "canonical_glm",
        {"t_r": 1.0, "confounds": ["trans_x"]},
    )
    context = TaskRunContext(
        subject="001",
        task="nback",
        run="1",
        bold_path=bold_path,
        events_path=events_path,
        confounds_path=confounds_path,
    )

    model = CanonicalGLMModel(spec)
    fitted = model.fit(spec.plan(context, str(tmp_path / "out")))
    statistics_path = model.write_sufficient_statistics(fitted)

    assert fitted.estimator.image == str(bold_path)
    assert list(fitted.events["trial_type"]) == ["one"]
    assert list(fitted.confounds.columns) == ["trans_x"]
    with np.load(statistics_path) as statistics:
        assert statistics["design_matrix"].shape == (1, 2)
        assert statistics["theta_0"].shape == (1, 2, 1)
        assert statistics["cov_0"].shape == (1, 2, 2)
    assert (statistics_path.parent / "sufficient_statistics.json").exists()


def test_activation_analysis_computes_contrasts_from_fitted_model(tmp_path: Path) -> None:
    class FakeImage:
        def to_filename(self, path: Path) -> None:
            path.write_bytes(b"contrast")

    class FakeEstimator:
        def compute_contrast(self, definition: str, *, output_type: str) -> FakeImage:
            assert definition == "two - one"
            assert output_type == "effect_size"
            return FakeImage()

    model = ModelSpec("canonical_glm", "canonical_glm", {})
    context = TaskRunContext(subject="001", task="nback", run="1")
    fitted = FittedModel(
        plan=model.plan(context, str(tmp_path / "out")),
        estimator=FakeEstimator(),  # type: ignore[arg-type]
        events=pd.DataFrame(),
        confounds=None,
    )
    analysis = AnalysisSpec(
        "activation",
        "canonical_glm",
        {"contrasts": {"two_gt_one": "two - one"}, "output_types": ["effect_size"]},
    )
    outputs = ActivationAnalysis().run(
        fitted,
        analysis.plan(context, model, str(tmp_path / "out")),
    )

    assert len(outputs) == 1
    assert outputs[0].path == (
        tmp_path / "out" / "sub-001" / "func" / "task-nback" / "run-1" / "analyses"
        / "activation" / "contrast-two_gt_one_stat-effect_size.nii.gz"
    )
    assert outputs[0].path.read_bytes() == b"contrast"


def test_canonical_glm_rejects_missing_required_confounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events_path = tmp_path / "events.tsv"
    confounds_path = tmp_path / "confounds.tsv"
    bold_path = tmp_path / "bold.nii.gz"
    bold_path.touch()
    events_path.write_text("onset\tduration\ttrial_type\n0\t1\tone\n", encoding="utf-8")
    confounds_path.write_text("trans_x\n0.1\n", encoding="utf-8")
    monkeypatch.setattr("pipeline.processing.analysis.models.canonical_glm.FirstLevelModel", lambda **kwargs: None)
    spec = ModelSpec("canonical_glm", "canonical_glm", {"confounds": ["rot_y"]})
    context = TaskRunContext(
        subject="001", task="nback", run="1", bold_path=bold_path,
        events_path=events_path, confounds_path=confounds_path,
    )

    with pytest.raises(ValueError, match="missing required columns: rot_y"):
        CanonicalGLMModel(spec).fit(spec.plan(context, str(tmp_path / "out")))


def test_atlas_cache_fetches_once_and_resamples_outside_bids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "atlas.nii.gz"
    target = tmp_path / "target.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)), target)
    calls: list[Path] = []

    class Fetched:
        maps = str(source)
        labels = ["one"]

    def fetch(**kwargs: object) -> Fetched:
        calls.append(Path(str(kwargs["data_dir"])))
        return Fetched()

    monkeypatch.setattr("pipeline.processing.analysis.analyses.atlas.fetch_atlas_schaefer_2018", fetch)
    cache_dir = tmp_path / "atlas-cache"
    cache = AtlasCache(cache_dir)

    first = cache.get("schaefer200")
    second = cache.get("schaefer200")
    output = tmp_path / "analysis" / "atlas-resampled.nii.gz"
    cache.resample("schaefer200", nib.load(target), output)

    assert first == second
    assert calls == [cache_dir]
    assert output.exists()
    assert not cache_dir.is_relative_to(tmp_path / "bids")
