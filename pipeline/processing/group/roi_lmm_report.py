from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


def render_roi_lmm_report(result: dict[str, Any], output_path: str | Path) -> Path:
    """Render a portable HTML report from serializable ROI LMM results."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    network_rows = []
    network_sections = []
    for network in result["networks"]:
        interaction = network["interaction"]
        network_rows.append({
            "network": network["network"],
            "n_subjects": network["n_subjects"],
            "random_effects": network["random_effects"],
            "estimate": interaction.get("estimate"),
            "t": interaction.get("statistic"),
            "p_value": interaction.get("p_value"),
            "fdr_q_value": interaction.get("fdr_q_value"),
            "fdr_significant": interaction.get("fdr_significant", False),
        })
        fixed_effects = pd.DataFrame.from_dict(network["fixed_effects"], orient="index").reset_index(
            names="effect"
        )
        fixed_effects = fixed_effects.rename(
            columns={
                "std_error": "SE",
                "lower_ci": "95% CI Lower",
                "upper_ci": "95% CI Upper",
                "statistic": "Statistic",
                "p_value": "P",
                "estimate": "Estimate",
            }
        )
        fixed_effects = fixed_effects[
            ["effect", "Estimate", "SE", "95% CI Lower", "95% CI Upper", "Statistic", "P"]
        ]
        decomposition = network["decomposition"]
        decomposition_table = pd.DataFrame(
            [
                {"Measure": "Control Change", "Estimate": decomposition["control_change"]},
                {"Measure": "Intervention Change", "Estimate": decomposition["intervention_change"]},
                {
                    "Measure": "Difference-in-Differences",
                    "Estimate": decomposition["difference_in_differences"],
                },
            ]
        )
        network_sections.append(
            f"<section><h2>{html.escape(network['network'])}</h2>"
            f"<p>Random-effects specification: <code>{html.escape(network['random_effects'])}</code>"
            + (
                f"; fallback reason: {html.escape(network['fallback_reason'])}"
                if network.get("fallback_reason")
                else ""
            )
            + "</p><h3>Fixed Effects</h3>"
            + fixed_effects.to_html(index=False)
            + "<h3>Estimated Marginal Means</h3>"
            + pd.DataFrame(
                [
                    {
                        "Group": row["group"].title(),
                        "Time": row["time"].title(),
                        "Estimate": row["estimate"],
                        "95% CI Lower": row["lower_ci"],
                        "95% CI Upper": row["upper_ci"],
                    }
                    for row in network["emmeans"]
                ]
            ).to_html(index=False)
            + "<h3>Interaction Decomposition</h3>"
            + decomposition_table.to_html(index=False)
            + (
                f'<h3>Interaction Plot</h3><p><img src="{html.escape(Path(network["plot"]).name)}" '
                f'alt="{html.escape(network["network"])} interaction plot"></p></section>'
            )
        )
    table = pd.DataFrame(network_rows).to_html(index=False) if network_rows else "<p>No estimable networks.</p>"
    content = (
        "<html><head><meta charset='utf-8'><title>ROI longitudinal mixed-effects analysis</title></head><body>"
        f"<h1>ROI longitudinal mixed-effects analysis</h1>"
        f"<p>Contrast: {html.escape(result['contrast'])}; atlas: {html.escape(result['atlas'])}; "
        f"subjects: {result['n_subjects']}; networks: {result['n_networks']}</p>"
        f"<h2>Methods and Analysis Summary</h2>"
        f"<p>Formula: <code>{html.escape(result['model_formula'])}</code>; "
        f"group coding: control={result['group_coding']['control']}, intervention={result['group_coding']['intervention']}; "
        f"time coding: baseline={result['time_coding']['baseline']}, followup={result['time_coding']['followup']}.</p>"
        "<p>Fixed-effect statistics are Wald tests from statsmodels MixedLM. "
        "Reported p-values are the model's Wald-test p-values. These may differ from "
        "lmerTest Satterthwaite or Kenward-Roger approximations commonly reported in R.</p>"
        f"<h2>Network Interactions</h2>{table}"
        f"{''.join(network_sections)}"
        "</body></html>"
    )
    output.write_text(content, encoding="utf-8")
    return output
