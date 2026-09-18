from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

GROUP_CODES = {"control": -0.5, "intervention": 0.5}
TIME_CODES = {"baseline": -0.5, "followup": 0.5}


def code_factors(data: pd.DataFrame) -> pd.DataFrame:
    """Add the explicit centered codes used by the longitudinal model."""

    result = data.copy()
    result["group_code"] = result["group"].map(GROUP_CODES)
    result["time_code"] = result["time"].map(TIME_CODES)
    invalid = result[result[["group_code", "time_code"]].isna().any(axis=1)]
    if not invalid.empty:
        raise ValueError("group and time must use the configured control/intervention and baseline/followup levels.")
    return result


def fit_network_lmm(
    data: pd.DataFrame,
    *,
    random_slope_time: bool = True,
) -> dict[str, Any]:
    """Fit ``effect ~ group * time + (1 + time | subject)`` for one network."""

    required = {"subject", "group", "time", "effect"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Network model data is missing columns: {missing}.")
    coded = code_factors(data)
    if coded["subject"].nunique() < 2:
        raise ValueError("A network model requires at least two subjects.")

    formula = "effect ~ group_code * time_code"
    fit = None
    specification = "random_intercept"
    fallback_reason = ""
    if random_slope_time:
        try:
            candidate = smf.mixedlm(
                formula, coded, groups=coded["subject"], re_formula="~time_code"
            ).fit(reml=False, method="lbfgs", disp=False)
            if not bool(getattr(candidate, "converged", False)):
                raise ValueError("random-slope model did not converge")
            diagnostic = _random_slope_diagnostic(candidate)
            if diagnostic is not None:
                raise ValueError(diagnostic)
            fit = candidate
            specification = "random_slope_time"
        except Exception as exc:  # statsmodels raises several fit-specific exception types.
            fallback_reason = str(exc)
            logger.warning(
                "Rejecting random-slope ROI LMM and refitting random-intercept model: %s",
                fallback_reason,
            )
    if fit is None:
        fit = smf.mixedlm(
            formula, coded, groups=coded["subject"], re_formula="1"
        ).fit(reml=False, method="lbfgs", disp=False)
        if random_slope_time:
            specification = "random_intercept_fallback"

    fixed_effects = _fixed_effects(fit)
    emmeans = _estimated_marginal_means(fit)
    decomposition = _decomposition(emmeans)
    interaction = fixed_effects.get("group_code:time_code", {})
    return {
        "n_observations": int(len(coded)),
        "n_subjects": int(coded["subject"].nunique()),
        "model_formula": formula,
        "group_coding": GROUP_CODES,
        "time_coding": TIME_CODES,
        "random_effects": specification,
        "fallback_reason": fallback_reason,
        "converged": bool(getattr(fit, "converged", False)),
        "fixed_effects": fixed_effects,
        "interaction": interaction,
        "emmeans": emmeans,
        "decomposition": decomposition,
    }


def apply_interaction_fdr(results: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Apply BH FDR to interaction p-values across networks."""

    valid = [(index, result["interaction"].get("p_value")) for index, result in enumerate(results)]
    valid = [(index, float(p_value)) for index, p_value in valid if p_value is not None and np.isfinite(p_value)]
    if not valid:
        return results
    rejected, corrected, _, _ = multipletests(
        [p_value for _, p_value in valid], alpha=alpha, method="fdr_bh"
    )
    for (index, _), q_value, is_rejected in zip(valid, corrected, rejected):
        results[index]["interaction"]["fdr_q_value"] = float(q_value)
        results[index]["interaction"]["fdr_significant"] = bool(is_rejected)
    return results


def _fixed_effects(fit: Any) -> dict[str, dict[str, float | None]]:
    names = list(fit.fe_params.index)
    effects = {}
    for name in names:
        estimate = float(fit.fe_params[name])
        standard_error = float(fit.bse_fe[name])
        effects[name] = {
            "estimate": estimate,
            "std_error": standard_error,
            "statistic": float(fit.tvalues[name]),
            "p_value": float(fit.pvalues[name]),
            "lower_ci": estimate - 1.96 * standard_error,
            "upper_ci": estimate + 1.96 * standard_error,
        }
    return effects


def _random_slope_diagnostic(fit: Any) -> str | None:
    """Return a rejection reason for a degenerate random-slope covariance."""

    covariance = np.asarray(fit.cov_re, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] < 2 or not np.isfinite(covariance).all():
        return "random-effects covariance is missing, non-finite, or degenerate"
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) <= 1e-8 * scale:
        return "random-effects covariance is singular or near-singular"
    names = list(getattr(fit.cov_re, "index", []))
    slope_index = next((index for index, name in enumerate(names) if str(name) == "time_code"), None)
    if slope_index is None:
        return "random-effects covariance has no time-slope variance"
    slope_variance = float(covariance[slope_index, slope_index])
    residual_scale = max(1.0, float(getattr(fit, "scale", 1.0)))
    if slope_variance <= 1e-8 * residual_scale:
        return "time random-slope variance is effectively zero"
    return None


def _estimated_marginal_means(fit: Any) -> list[dict[str, float | str]]:
    names = list(fit.fe_params.index)
    covariance = fit.cov_params().loc[names, names].to_numpy()
    rows = []
    for group, group_code in GROUP_CODES.items():
        for time, time_code in TIME_CODES.items():
            vector = np.array([1.0, group_code, time_code, group_code * time_code])
            vector = vector[: len(names)]
            estimate = float(vector @ fit.fe_params.to_numpy())
            standard_error = float(np.sqrt(max(vector @ covariance @ vector, 0.0)))
            rows.append({
                "group": group,
                "time": time,
                "estimate": estimate,
                "std_error": standard_error,
                "lower_ci": estimate - 1.96 * standard_error,
                "upper_ci": estimate + 1.96 * standard_error,
            })
    return rows


def _decomposition(emmeans: list[dict[str, float | str]]) -> dict[str, float]:
    values = {(row["group"], row["time"]): float(row["estimate"]) for row in emmeans}
    control_change = values[("control", "followup")] - values[("control", "baseline")]
    intervention_change = values[("intervention", "followup")] - values[("intervention", "baseline")]
    return {
        "control_change": control_change,
        "intervention_change": intervention_change,
        "difference_in_differences": intervention_change - control_change,
    }
