import json
from pathlib import Path

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
from pipeline.processing.analysis.derivatives import DerivativePathBuilder
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


def test_analysis_service_only_plans_and_does_not_execute(tmp_path: Path) -> None:
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
    assert "no model execution" in result["message"]
    assert not (tmp_path / "derivatives").exists()


def test_derivative_path_builder_rejects_namespace_path_traversal() -> None:
    context = TaskRunContext(subject="001", task="nback")

    with pytest.raises(ValueError, match="single path components"):
        DerivativePathBuilder.for_context("derivatives", context, namespace="model", name="../model")
