import pandas as pd
import pytest

from pipeline.config import _apply_defaults, validate_config
from pipeline.processing.group.roi_lmm_model import (
    apply_interaction_fdr,
    code_factors,
    fit_network_lmm,
)


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


def test_network_lmm_returns_emmeans_and_difference_in_differences() -> None:
    result = fit_network_lmm(_synthetic_network(), random_slope_time=False)

    assert result["random_effects"] == "random_intercept"
    assert len(result["emmeans"]) == 4
    assert result["decomposition"]["difference_in_differences"] > 1.0
    assert result["interaction"]["p_value"] < 0.05
    assert {
        "estimate", "std_error", "statistic", "p_value", "lower_ci", "upper_ci"
    }.issubset(result["fixed_effects"]["group_code"])


def test_interaction_fdr_is_applied_across_networks() -> None:
    results = [
        {"interaction": {"p_value": 0.001}},
        {"interaction": {"p_value": 0.20}},
    ]

    apply_interaction_fdr(results)

    assert results[0]["interaction"]["fdr_q_value"] < 0.01
    assert results[0]["interaction"]["fdr_significant"] is True
    assert results[1]["interaction"]["fdr_q_value"] > 0.1


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
