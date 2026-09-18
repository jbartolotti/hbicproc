from __future__ import annotations

import logging
import warnings
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
    network_name: str = "unknown",
) -> dict[str, Any]:
    """Fit ``effect ~ group * time + (1 + time | subject)`` for one ROI unit."""

    required = {"subject", "group", "time", "effect"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Network model data is missing columns: {missing}.")
    coded = code_factors(data)
    if coded["subject"].nunique() < 2:
        raise ValueError("A network model requires at least two subjects.")

    formula = "effect ~ group_code * time_code"
    intercept_model = None
    explicit_intercept_model = None
    for model_label, re_formula in (
        ("random_intercept_default", None),
        ("random_intercept_re_formula_1", "1"),
    ):
        try:
            model_kwargs = {"groups": coded["subject"]}
            if re_formula is not None:
                model_kwargs["re_formula"] = re_formula
            model = smf.mixedlm(formula, coded, **model_kwargs)
        except Exception as exc:
            logger.debug(
                "ROI LMM model structure construction failed: network=%s model=%s reason=%s",
                network_name,
                model_label,
                exc,
            )
            continue
        _log_model_structure(
            model,
            network_name=network_name,
            model_label=model_label,
            n_observations=len(coded),
            n_subjects=coded["subject"].nunique(),
        )
        if re_formula is None:
            intercept_model = model
        else:
            explicit_intercept_model = model
    fit_warning = ""
    random_slope_diagnostic = ""
    random_intercept_diagnostic = ""
    cov_re_summary: dict[str, Any] = {}
    fit = None
    specification = "random_intercept"
    fallback_reason = ""
    if random_slope_time:
        try:
            candidate, slope_warnings = _fit_model(
                smf.mixedlm(formula, coded, groups=coded["subject"], re_formula="~time_code"),
                model_type="random_slope_time",
            )
            fit_warning = slope_warnings
            slope_diagnostics = _log_fit_diagnostics(
                candidate,
                network_name=network_name,
                model_type="random_slope_time",
                n_observations=len(coded),
                n_subjects=coded["subject"].nunique(),
            )
            cov_re_summary = slope_diagnostics["cov_re_summary"]
            if not bool(getattr(candidate, "converged", False)):
                raise ValueError("random-slope model did not converge")
            diagnostic = _random_slope_diagnostic(candidate)
            if diagnostic is not None:
                raise ValueError(diagnostic)
            fit = candidate
            specification = "random_slope_time"
        except Exception as exc:  # statsmodels raises several fit-specific exception types.
            fallback_reason = str(exc)
            random_slope_diagnostic = fallback_reason
            logger.warning(
                "Network '%s': rejecting random-slope ROI LMM and refitting random-intercept model: %s",
                network_name,
                fallback_reason,
            )
    if fit is None:
        try:
            if explicit_intercept_model is None:
                explicit_intercept_model = smf.mixedlm(
                    formula, coded, groups=coded["subject"], re_formula="1"
                )
            fit, intercept_warnings = _fit_model(
                explicit_intercept_model,
                model_type="random_intercept",
            )
            fit_warning = "; ".join(filter(None, [fit_warning, intercept_warnings]))
        except Exception as exc:
            random_intercept_diagnostic = str(exc)
            fallback_reason = "; ".join(
                filter(
                    None,
                    [
                        fallback_reason,
                        f"random-intercept model failed: {random_intercept_diagnostic}",
                    ],
                )
            )
            logger.warning(
                "Network '%s': random-intercept ROI LMM failed after diagnostics: %s",
                network_name,
                random_intercept_diagnostic,
            )
            fit, ols_warnings = _fit_model(
                smf.ols(formula, data=coded),
                model_type="ols",
            )
            fit_warning = "; ".join(filter(None, [fit_warning, ols_warnings]))
            specification = "ols_fallback"
            cov_re_summary = {}
        else:
            intercept_diagnostics = _log_fit_diagnostics(
                fit,
                network_name=network_name,
                model_type="random_intercept",
                n_observations=len(coded),
                n_subjects=coded["subject"].nunique(),
            )
            cov_re_summary = intercept_diagnostics["cov_re_summary"]
            random_intercept_diagnostic = _random_intercept_diagnostic(fit) or ""
            if random_slope_time:
                specification = "random_intercept_fallback"

    if specification == "ols_fallback":
        _log_fit_diagnostics(
            fit,
            network_name=network_name,
            model_type="ols",
            n_observations=len(coded),
            n_subjects=coded["subject"].nunique(),
        )

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
        "random_slope_diagnostic": random_slope_diagnostic,
        "random_intercept_diagnostic": random_intercept_diagnostic,
        "model_type": specification,
        "cov_re_summary": cov_re_summary,
        "fit_warning": fit_warning,
        "converged": bool(getattr(fit, "converged", True)),
        "fixed_effects": fixed_effects,
        "interaction": interaction,
        "emmeans": emmeans,
        "decomposition": decomposition,
    }


def apply_interaction_fdr(results: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Apply BH FDR to interaction p-values across ROI units."""

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
    parameters = _fixed_parameters(fit)
    names = list(parameters.index)
    standard_errors = getattr(fit, "bse_fe", getattr(fit, "bse", None))
    effects = {}
    for name in names:
        estimate = float(parameters[name])
        standard_error = float(standard_errors[name])
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


def _log_fit_diagnostics(
    fit: Any,
    *,
    network_name: str,
    model_type: str,
    n_observations: int,
    n_subjects: int,
) -> dict[str, Any]:
    """Log and serialize diagnostics for a completed statsmodels fit."""

    random_covariance = getattr(fit, "cov_re", None)
    covariance = np.asarray(random_covariance, dtype=float) if random_covariance is not None else np.array([])
    eigenvalues = np.linalg.eigvalsh(covariance) if covariance.ndim == 2 else np.array([])
    variances = np.diag(covariance).tolist() if covariance.ndim == 2 else []
    minimum_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size else None
    singularity_diagnostic = ""
    if minimum_eigenvalue is not None:
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if minimum_eigenvalue <= 1e-8 * scale:
            singularity_diagnostic = "covariance singular or near-singular"
    if variances and variances[0] <= 1e-8 * max(1.0, float(getattr(fit, "scale", 1.0))):
        singularity_diagnostic = "; ".join(
            filter(None, [singularity_diagnostic, "first random-effect variance effectively zero"])
        )
    summary = {
        "matrix": covariance.tolist(),
        "variance_estimates": variances,
        "minimum_eigenvalue": minimum_eigenvalue,
        "residual_variance": float(getattr(fit, "scale", np.nan)),
        "singularity_diagnostic": singularity_diagnostic,
    }
    logger.debug(
        "ROI LMM diagnostics: network=%s model=%s observations=%d subjects=%d "
        "converged=%s cov_re=%s variance_estimates=%s residual_variance=%s "
        "minimum_eigenvalue=%s fixed_effects=%s",
        network_name,
        model_type,
        n_observations,
        n_subjects,
        bool(getattr(fit, "converged", False)),
        covariance.tolist(),
        variances,
        getattr(fit, "scale", None),
        minimum_eigenvalue,
        {name: float(value) for name, value in _fixed_parameters(fit).items()},
    )
    if singularity_diagnostic:
        logger.debug(
            "ROI LMM singularity diagnostics: network=%s model=%s reason=%s",
            network_name,
            model_type,
            singularity_diagnostic,
        )
    return {"cov_re_summary": summary}


def _log_model_structure(
    model: Any,
    *,
    network_name: str,
    model_label: str,
    n_observations: int,
    n_subjects: int,
) -> None:
    """Log the design structures used by each random-intercept construction."""

    exog = np.asarray(model.exog)
    exog_re = np.asarray(model.exog_re) if model.exog_re is not None else np.empty((0, 0))
    fixed_rank = int(np.linalg.matrix_rank(exog)) if exog.size else 0
    condition_number = float(np.linalg.cond(exog)) if exog.size else None
    random_condition_number = float(np.linalg.cond(exog_re)) if exog_re.size else None
    logger.debug(
        "ROI LMM model structure: network=%s model=%s exog_shape=%s "
        "exog_re_shape=%s fixed_effect_rank=%d condition_number=%s "
        "random_effect_condition_number=%s subjects=%d observations=%d",
        network_name,
        model_label,
        exog.shape,
        exog_re.shape,
        fixed_rank,
        condition_number,
        random_condition_number,
        n_subjects,
        n_observations,
    )


def _fit_model(model: Any, *, model_type: str) -> tuple[Any, str]:
    """Fit without hiding warnings; return their text for report diagnostics."""

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if model_type == "ols":
            fit = model.fit()
        else:
            fit = model.fit(reml=False, method="lbfgs", disp=False)
    messages = []
    for warning in captured:
        warnings.warn(warning.message, warning.category, stacklevel=2)
        messages.append(str(warning.message))
    return fit, "; ".join(dict.fromkeys(messages))


def _fixed_parameters(fit: Any) -> Any:
    return getattr(fit, "fe_params", getattr(fit, "params"))


def _random_intercept_diagnostic(fit: Any) -> str | None:
    summary = _log_covariance_summary(fit)
    minimum_eigenvalue = summary["minimum_eigenvalue"]
    variances = summary["variance_estimates"]
    if minimum_eigenvalue is not None and minimum_eigenvalue <= 1e-8:
        return "random-intercept covariance is singular or near-singular"
    if variances and variances[0] <= 1e-8 * max(1.0, summary["residual_variance"]):
        return "subject random-intercept variance is effectively zero"
    return None


def _log_covariance_summary(fit: Any) -> dict[str, Any]:
    random_covariance = getattr(fit, "cov_re", None)
    covariance = np.asarray(random_covariance, dtype=float) if random_covariance is not None else np.array([])
    eigenvalues = np.linalg.eigvalsh(covariance) if covariance.ndim == 2 else np.array([])
    return {
        "variance_estimates": np.diag(covariance).tolist() if covariance.ndim == 2 else [],
        "minimum_eigenvalue": float(np.min(eigenvalues)) if eigenvalues.size else None,
        "residual_variance": float(getattr(fit, "scale", np.nan)),
    }


def _estimated_marginal_means(fit: Any) -> list[dict[str, float | str]]:
    parameters = _fixed_parameters(fit)
    names = list(parameters.index)
    covariance = fit.cov_params().loc[names, names].to_numpy()
    rows = []
    for group, group_code in GROUP_CODES.items():
        for time, time_code in TIME_CODES.items():
            vector = np.array([1.0, group_code, time_code, group_code * time_code])
            vector = vector[: len(names)]
            estimate = float(vector @ parameters.to_numpy())
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
