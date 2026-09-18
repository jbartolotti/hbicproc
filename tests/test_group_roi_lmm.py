import pandas as pd
import pytest
from pathlib import Path

from pipeline.config import _apply_defaults, validate_config
from pipeline.processing.group.roi_lmm_model import (
    apply_interaction_fdr,
    code_factors,
    fit_network_lmm,
)
from pipeline.processing.group.roi_lmm_report import render_roi_lmm_report

from pipeline.processing.group.voxelwise_lme import build_3dlmer_command, build_afni_data_table

def _synthetic_network() -> pd.DataFrame:
    rows = []
    for subject in range(1, 9):
        group = "control" if subject <= 4 else "intervention"
        subject_offset = subject * 0.01
        for time, time_offset in [("baseline", 0.0), ("followup", 1.0)]:
            effect = subject_offset + time_offset
            if group == "intervention":
                effect += 2.0 * time_offset
            rows.append({"subject": f"{subject:03d}", "group": group, "time": time, "effect": effect})
    return pd.DataFrame(rows)


def test_code_factors_uses_centered_half_coding() -> None:
    coded = code_factors(
        pd.DataFrame(
            [
                {"subject": "001", "group": "control", "time": "baseline", "effect": 0.0},
                {"subject": "002", "group": "intervention", "time": "followup", "effect": 1.0},
            ]
        )
    )

    assert coded["group_code"].tolist() == [-0.5, 0.5]
    assert coded["time_code"].tolist() == [-0.5, 0.5]


def test_network_lmm_returns_diagnostics_and_difference_in_differences(caplog) -> None:
    with caplog.at_level("DEBUG"):
        result = fit_network_lmm(
            _synthetic_network(), random_slope_time=False, network_name="Default"
        )

    assert result["random_effects"] == "random_intercept"
    assert len(result["emmeans"]) == 4
    assert result["decomposition"]["difference_in_differences"] > 1.0
    assert result["interaction"]["p_value"] < 0.05
    assert {
        "estimate", "std_error", "statistic", "p_value", "lower_ci", "upper_ci"
    }.issubset(result["fixed_effects"]["group_code"])
    assert result["model_type"] == "random_intercept"
    assert result["cov_re_summary"]["matrix"]
    assert "residual_variance" in result["cov_re_summary"]
    assert "minimum_eigenvalue" in result["cov_re_summary"]
    assert any("network=Default" in record.message for record in caplog.records)
    assert any("cov_re=" in record.message for record in caplog.records)
    structure_logs = [
        record.message for record in caplog.records if "ROI LMM model structure" in record.message
    ]
    assert any("model=random_intercept_default" in message for message in structure_logs)
    assert any("model=random_intercept_re_formula_1" in message for message in structure_logs)
    assert all("exog_shape=" in message for message in structure_logs)
    assert all("exog_re_shape=" in message for message in structure_logs)
    assert all("fixed_effect_rank=" in message for message in structure_logs)
    assert all("condition_number=" in message for message in structure_logs)


def test_interaction_fdr_is_applied_across_networks() -> None:
    results = [
        {"interaction": {"p_value": 0.001}},
        {"interaction": {"p_value": 0.20}},
    ]

    apply_interaction_fdr(results)

    assert results[0]["interaction"]["fdr_q_value"] < 0.01
    assert results[0]["interaction"]["fdr_significant"] is True
    assert results[1]["interaction"]["fdr_q_value"] > 0.1


def test_network_lmm_uses_ols_when_both_mixed_models_fail(monkeypatch) -> None:
    def fail_mixed_model(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    import numpy as np

    monkeypatch.setattr(
        "pipeline.processing.group.roi_lmm_model.smf.mixedlm", fail_mixed_model
    )

    result = fit_network_lmm(_synthetic_network(), network_name="Default")

    assert result["model_type"] == "ols_fallback"
    assert result["random_intercept_diagnostic"] == "Singular matrix"
    assert "random-intercept model failed: Singular matrix" in result["fallback_reason"]
    assert result["fixed_effects"]["group_code:time_code"]["p_value"] < 0.05


def test_roi_lmm_report_includes_model_diagnostics_and_wald_note(tmp_path: Path) -> None:
    network = fit_network_lmm(_synthetic_network(), random_slope_time=False, network_name="Default")
    network["network"] = "Default"
    plot_path = tmp_path / "default.png"
    plot_path.write_bytes(b"png")
    network["plot"] = str(plot_path)
    report = render_roi_lmm_report(
        {
            "contrast": "memory",
            "atlas": "schaefer200",
            "n_subjects": network["n_subjects"],
            "n_networks": 1,
            "model_formula": network["model_formula"],
            "group_coding": network["group_coding"],
            "time_coding": network["time_coding"],
            "networks": [network],
        },
        tmp_path / "report.html",
    )

    content = report.read_text(encoding="utf-8")
    assert "Wald tests from statsmodels MixedLM" in content
    assert "Model Diagnostics" in content
    assert "95% CI Lower" in content
    assert "default.png" in content


def test_roi_lmm_configuration_requires_analysis_inputs() -> None:
    config = _apply_defaults({
        "group": {
            "roi_lmm": {
                "enabled": True,
                "task": "nback",
                "atlas": "schaefer200",
                "contrasts": ["memory"],
                "factors": {"group": {}, "time": {}},
            }
        }
    })
    validate_config(config)

    invalid = _apply_defaults({"group": {"roi_lmm": {"enabled": True}}})
    with pytest.raises(ValueError, match="task"):
        validate_config(invalid)


def test_voxelwise_lme_builds_afni_table_and_command(tmp_path) -> None:
    records = pd.DataFrame([
        {"subject": "001", "session": "BL", "path": tmp_path / "sub-001_BL.nii.gz"},
        {"subject": "001", "session": "W12", "path": tmp_path / "sub-001_W12.nii.gz"},
    ])
    participants = pd.DataFrame([{"subject": "001", "group": "control"}])

    table = build_afni_data_table(records, participants)
    command = build_3dlmer_command(
        output_prefix=tmp_path / "3dLMEr",
        data_table=tmp_path / "afni_data_table.tsv",
        mask=tmp_path / "mask.nii.gz",
    )

    assert list(table.columns) == ["Subj", "Group", "Time", "InputFile"]
    assert table["Time"].tolist() == ["baseline", "followup"]
    assert "-model" in command and "Group*Time" in command
    assert "-ranEff" in command and "~1|Subj" in command
    assert "-dataTable" in command and "@" in command[command.index("-dataTable") + 1]
    assert "GroupXTime" in command


def test_voxelwise_lme_configuration_is_simple_and_afni_only() -> None:
    config = _apply_defaults({
        "group": {
            "voxelwise_lme": {
                "enabled": True,
                "engine": "afni",
                "contrasts": ["memory"],
                "mask": {"source": "template_gm", "gm_probability_threshold": 0.2},
            }
        }
    })
    validate_config(config)

    invalid = _apply_defaults({
        "group": {
            "voxelwise_lme": {
                "enabled": True,
                "engine": "python",
                "contrasts": ["memory"],
            }
        }
    })
    with pytest.raises(ValueError, match="engine"):
        validate_config(invalid)
