from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from ..base import AnalysisPlugin, AnalysisResult
from ..dataset import DatasetIndex
from ..derivatives import DerivativePathBuilder

logger = logging.getLogger(__name__)


def _normalize_entity_value(value: Any, *, entity_name: str) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    for candidate_prefix in (f"{entity_name}-", f"{entity_name}_", entity_name):
        if text.lower().startswith(candidate_prefix.lower()):
            text = text[len(candidate_prefix) :]
            break

    return text.replace("_", "-")


class ActivationAnalysisPlugin(AnalysisPlugin):
    name = "activation"
    description = "Subject-level task-fMRI activation analysis with Nilearn."

    MOTION_COLUMNS = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
    ]
    MOTION_DERIVATIVE_COLUMNS = [
        "trans_x_derivative1",
        "trans_y_derivative1",
        "trans_z_derivative1",
        "rot_x_derivative1",
        "rot_y_derivative1",
        "rot_z_derivative1",
    ]

    def run(
        self,
        subject: str,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        rerun: bool = False,
        plugin_config: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        normalized_subject = _normalize_entity_value(subject, entity_name="sub")
        logger.info("Activation analysis requested for subject=%s", normalized_subject)

        plugin_cfg = plugin_config or self.get_config(config)
        if not isinstance(plugin_cfg, dict):
            plugin_cfg = {}

        tasks = plugin_cfg.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = [str(task).strip() for task in tasks if str(task).strip()]

        if not tasks:
            return AnalysisResult(
                success=True,
                skipped=True,
                plugin=self.name,
                message=(
                    f"Activation analysis for subject {normalized_subject} is not configured with any tasks. "
                    "Set analysis.subject.activation.tasks to the tasks to analyze."
                ),
                details={"subject": normalized_subject, "tasks": tasks},
            )

        try:
            self._validate_contrast_config(plugin_cfg)
        except ValueError as exc:
            return AnalysisResult(
                success=False,
                plugin=self.name,
                message=str(exc),
                details={"subject": normalized_subject, "tasks": tasks, "contrasts": plugin_cfg.get("contrasts")},
            )

        results: list[AnalysisResult] = []
        for task in tasks:
            logger.info("Starting activation task for subject=%s task=%s", normalized_subject, task)
            results.extend(self._run_task(normalized_subject, task, config, plugin_cfg, dry_run=dry_run))

        if not results:
            return AnalysisResult(
                success=False,
                plugin=self.name,
                message=f"No activation analysis runs were found for subject {normalized_subject}.",
                details={"subject": normalized_subject, "tasks": tasks},
            )

        success = all(result.success for result in results)
        message = "\n".join(result.message for result in results)
        return AnalysisResult(
            success=success,
            plugin=self.name,
            message=message,
            details={
                "subject": normalized_subject,
                "tasks": tasks,
                "results": [result.details for result in results],
            },
        )

    def _validate_contrast_config(self, plugin_cfg: dict[str, Any]) -> dict[str, str]:
        explicit = plugin_cfg.get("contrasts")
        if isinstance(explicit, dict):
            normalized = {str(name): str(expression) for name, expression in explicit.items() if str(name).strip() and str(expression).strip()}
            if normalized:
                return normalized
        if isinstance(explicit, list):
            normalized = {str(item): str(item) for item in explicit if str(item).strip()}
            if normalized:
                return normalized
        raise ValueError(
            "Activation analysis requires explicit 'analysis.subject.activation.contrasts' definitions. "
            "No valid contrast configuration was provided."
        )

    def _run_task(
        self,
        subject: str,
        task: str,
        config: dict[str, Any],
        plugin_cfg: dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> list[AnalysisResult]:
        dataset = DatasetIndex.from_config(config)
        run_infos = [run.as_dict() for run in dataset.get_task_runs(subject=subject, task=task)]
        if not run_infos:
            return [
                AnalysisResult(
                    success=False,
                    plugin=self.name,
                    message=(
                        f"No BOLD, events, and confounds files were found for subject {subject} "
                        f"task {task}."
                    ),
                    details={"subject": subject, "task": task},
                )
            ]

        results: list[AnalysisResult] = []
        for run_info in run_infos:
            try:
                result = self._run_single_run(subject, task, run_info, config, plugin_cfg, dry_run=dry_run)
            except Exception as exc:  # pragma: no cover - runtime failure path; surfaced to caller
                results.append(
                    AnalysisResult(
                        success=False,
                        plugin=self.name,
                        message=f"Activation analysis failed for {subject} task {task}: {exc}",
                        details={"subject": subject, "task": task, "run": run_info.get("run"), "error": str(exc)},
                    )
                )
            else:
                results.append(result)
        return results

    def _run_single_run(
        self,
        subject: str,
        task: str,
        run_info: dict[str, Any],
        config: dict[str, Any],
        plugin_cfg: dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> AnalysisResult:
        from nilearn.glm.first_level import FirstLevelModel

        normalized_subject = _normalize_entity_value(subject, entity_name="sub")
        session_label = _normalize_entity_value(run_info.get("session"), entity_name="ses")
        run_label = _normalize_entity_value(run_info.get("run"), entity_name="run") or "1"

        logger.info(
            "Running activation analysis for subject=%s session=%s task=%s run=%s",
            normalized_subject,
            session_label or "unspecified",
            task,
            run_label,
        )

        output_root = self._resolve_output_root(config, plugin_cfg)
        output_dir = output_root / normalized_subject
        if session_label:
            output_dir = output_dir / session_label
        output_dir = output_dir / "func"
        output_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            return AnalysisResult(
                success=True,
                skipped=True,
                plugin=self.name,
                message=(
                    f"Dry run: would run activation analysis for subject {normalized_subject} task {task} "
                    f"with session {session_label or 'unspecified'} run {run_label} in {output_dir}."
                ),
                details={
                    "subject": normalized_subject,
                    "task": task,
                    "session": session_label,
                    "run": run_label,
                    "output_dir": str(output_dir),
                },
            )

        events = pd.read_csv(run_info["events_path"], sep="\t")
        if events.empty:
            raise ValueError(f"Event file for {normalized_subject} task {task} is empty: {run_info['events_path']}")

        events = events.copy()
        if "trial_type" in events.columns:
            events["trial_type"] = events["trial_type"].astype(str).str.strip().str.lower().str.replace(" ", "_")

        confounds = pd.read_csv(run_info["confounds_path"], sep="\t")
        confounds = self._prepare_confounds(confounds, plugin_cfg)

        tr = self._read_repetition_time(run_info["bold_path"])
        logger.info(
            "Fitting activation model for subject=%s session=%s task=%s run=%s TR=%s",
            normalized_subject,
            session_label or "unspecified",
            task,
            run_label,
            tr,
        )
        glm = FirstLevelModel(
            t_r=float(tr),
            hrf_model=str(plugin_cfg.get("hrf_model", "spm")),
            drift_model=str(plugin_cfg.get("drift_model", "cosine")),
            high_pass=float(plugin_cfg.get("high_pass", 0.01)),
            smoothing_fwhm=float(plugin_cfg.get("smoothing_fwhm", 6.0)),
            minimize_memory=bool(plugin_cfg.get("minimize_memory", False)),
        )
        glm.fit(run_info["bold_path"], events=events, confounds=confounds if not confounds.empty else None)

        logger.info(
            "Generating activation contrasts for subject=%s session=%s task=%s run=%s",
            normalized_subject,
            session_label or "unspecified",
            task,
            run_label,
        )
        contrast_map = self._build_contrast_map(plugin_cfg)
        design_matrix = glm.design_matrices_[0]

        design_matrix_path = self._output_path(
            output_dir,
            subject=normalized_subject,
            session=session_label,
            task=task,
            run=run_label,
            desc="design_matrix",
            suffix=".csv",
        )
        logger.info("Saving design matrix CSV for subject=%s session=%s task=%s run=%s", normalized_subject, session_label or "unspecified", task, run_label)
        design_matrix.to_csv(design_matrix_path, index=False)

        design_png_path = self._output_path(
            output_dir,
            subject=normalized_subject,
            session=session_label,
            task=task,
            run=run_label,
            desc="design_matrix",
            suffix=".png",
        )
        logger.info("Saving design matrix PNG for subject=%s session=%s task=%s run=%s", normalized_subject, session_label or "unspecified", task, run_label)
        self._save_design_png(design_matrix, design_png_path)

        fd_threshold = float(
            plugin_cfg.get("fd_threshold", plugin_cfg.get("motion_qc", {}).get("fd_threshold", 0.5))
        )
        motion_qc = self._summarize_motion_qc(confounds, fd_threshold)
        motion_qc_path = self._output_path(
            output_dir,
            subject=normalized_subject,
            session=session_label,
            task=task,
            run=run_label,
            desc="motion_qc",
            suffix=".json",
        )
        logger.info("Saving motion QC for subject=%s session=%s task=%s run=%s", normalized_subject, session_label or "unspecified", task, run_label)
        motion_qc_path.write_text(json.dumps(motion_qc, indent=2), encoding="utf-8")

        generated_paths = [
            str(design_matrix_path),
            str(design_png_path),
            str(motion_qc_path),
        ]

        for contrast_name, expression in contrast_map.items():
            logger.info(
                "Saving contrast outputs for subject=%s session=%s task=%s run=%s contrast=%s",
                normalized_subject,
                session_label or "unspecified",
                task,
                run_label,
                contrast_name,
            )
            effect_map = glm.compute_contrast(expression, output_type="effect_size")
            z_map = glm.compute_contrast(expression, output_type="z_score")
            variance_map = glm.compute_contrast(expression, output_type="effect_variance")

            effect_path = self._output_path(
                output_dir,
                subject=normalized_subject,
                session=session_label,
                task=task,
                run=run_label,
                desc=f"{contrast_name}",
                stat="effect",
                suffix=".nii.gz",
            )
            z_path = self._output_path(
                output_dir,
                subject=normalized_subject,
                session=session_label,
                task=task,
                run=run_label,
                desc=f"{contrast_name}",
                stat="z",
                suffix=".nii.gz",
            )
            variance_path = self._output_path(
                output_dir,
                subject=normalized_subject,
                session=session_label,
                task=task,
                run=run_label,
                desc=f"{contrast_name}",
                stat="variance",
                suffix=".nii.gz",
            )
            effect_map.to_filename(effect_path)
            z_map.to_filename(z_path)
            variance_map.to_filename(variance_path)
            generated_paths.extend([str(effect_path), str(z_path), str(variance_path)])

        for condition in self._infer_conditions(events, plugin_cfg):
            try:
                logger.info(
                    "Saving condition effect map for subject=%s session=%s task=%s run=%s condition=%s",
                    normalized_subject,
                    session_label or "unspecified",
                    task,
                    run_label,
                    condition,
                )
                effect_map = glm.compute_contrast(condition, output_type="effect_size")
            except Exception:
                continue
            effect_path = self._output_path(
                output_dir,
                subject=normalized_subject,
                session=session_label,
                task=task,
                run=run_label,
                desc=condition,
                stat="effect",
                suffix=".nii.gz",
            )
            effect_map.to_filename(effect_path)
            generated_paths.append(str(effect_path))

        residuals = getattr(glm, "residuals_", None)
        if residuals is None:
            residuals = getattr(glm, "residuals", None)
        if residuals is None:
            logger.warning(
                "Nilearn did not expose residual images for subject %s task %s; continuing without residual outputs.",
                normalized_subject,
                task,
            )
        else:
            residual_img = residuals[0] if isinstance(residuals, (list, tuple)) and residuals else residuals
            residual_path = self._output_path(
                output_dir,
                subject=normalized_subject,
                session=session_label,
                task=task,
                run=run_label,
                desc="residual",
                suffix=".nii.gz",
            )
            logger.info("Saving residual image for subject=%s session=%s task=%s run=%s", normalized_subject, session_label or "unspecified", task, run_label)
            residual_img.to_filename(residual_path)
            generated_paths.append(str(residual_path))

        mask_img = getattr(getattr(glm, "masker_", None), "mask_img_", None)
        if mask_img is not None:
            mask_path = self._output_path(
                output_dir,
                subject=normalized_subject,
                session=session_label,
                task=task,
                run=run_label,
                desc="mask",
                suffix=".nii.gz",
            )
            logger.info("Saving mask image for subject=%s session=%s task=%s run=%s", normalized_subject, session_label or "unspecified", task, run_label)
            mask_img.to_filename(mask_path)
            generated_paths.append(str(mask_path))

        report_path = self._output_path(
            output_dir,
            subject=normalized_subject,
            session=session_label,
            task=task,
            run=run_label,
            desc="report",
            suffix=".html",
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(
                "Generating Nilearn HTML report for subject=%s session=%s task=%s run=%s",
                normalized_subject,
                session_label or "unspecified",
                task,
                run_label,
            )
            report = glm.generate_report(contrasts=contrast_map)
            report.save_as_html(report_path)
            generated_paths.append(str(report_path))
        except Exception:
            logger.warning(
                "Failed to generate HTML report for subject=%s session=%s task=%s run=%s; scientific outputs were preserved.",
                normalized_subject,
                session_label or "unspecified",
                task,
                run_label,
                exc_info=True,
            )

        summary = {
            "subject": normalized_subject,
            "task": task,
            "session": session_label,
            "run": run_label,
            "output_dir": str(output_dir),
            "motion_qc": motion_qc,
            "generated_outputs": generated_paths,
            "report_generated": str(report_path) in generated_paths,
        }
        return AnalysisResult(
            success=True,
            plugin=self.name,
            message=(
                f"Activation analysis complete for subject {normalized_subject} task {task} "
                f"session {session_label or 'unspecified'} run {run_label}."
            ),
            details=summary,
        )

    def _resolve_output_root(self, config: dict[str, Any], plugin_cfg: dict[str, Any]) -> Path:
        output_value = plugin_cfg.get("output_dir") or "derivatives/hbicproc"
        output_path = Path(output_value)
        if not output_path.is_absolute():
            output_path = Path(config.get("study_root", ".")) / output_path
        return output_path

    def _expand_confounds(self, confounds_spec: Any) -> list[str]:
        if confounds_spec is None:
            return []
        if isinstance(confounds_spec, list):
            return [str(item).strip() for item in confounds_spec if str(item).strip()]
        if not isinstance(confounds_spec, dict):
            return []

        expanded: list[str] = []
        if confounds_spec.get("motion"):
            expanded.extend(self.MOTION_COLUMNS)
        if confounds_spec.get("motion_derivatives"):
            expanded.extend(self.MOTION_DERIVATIVE_COLUMNS)
        if confounds_spec.get("framewise_displacement"):
            expanded.append("framewise_displacement")

        acompcor_count = confounds_spec.get("acompcor")
        if acompcor_count is not None:
            try:
                count = int(acompcor_count)
            except (TypeError, ValueError) as exc:  # pragma: no cover - validation failure path
                raise ValueError(f"Invalid a_comp_cor configuration value: {acompcor_count!r}") from exc
            expanded.extend(f"a_comp_cor_{index:02d}" for index in range(max(count, 0)))

        for key, value in confounds_spec.items():
            if key in {"motion", "motion_derivatives", "framewise_displacement", "acompcor"}:
                continue
            if value is True:
                expanded.append(str(key))
            elif isinstance(value, (list, tuple, set)):
                expanded.extend(str(item).strip() for item in value if str(item).strip())
        return list(dict.fromkeys(str(item).strip() for item in expanded if str(item).strip()))

    def _prepare_confounds(self, confounds: pd.DataFrame, plugin_cfg: dict[str, Any]) -> pd.DataFrame:
        if confounds.empty:
            return pd.DataFrame()

        configured_columns = self._expand_confounds(plugin_cfg.get("confounds"))
        if not configured_columns:
            return pd.DataFrame()
        available_columns = [column for column in configured_columns if column in confounds.columns]
        if not available_columns:
            return pd.DataFrame()
        return confounds.loc[:, available_columns].copy().fillna(0.0)

    def _read_repetition_time(self, bold_path: str | Path) -> float:
        bold_path = Path(bold_path)
        stem = bold_path.name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        json_candidates = [
            bold_path.with_name(f"{stem}.json"),
            bold_path.parent / f"{stem}.json",
        ]
        for candidate in json_candidates:
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            tr = metadata.get("RepetitionTime")
            if tr is not None:
                return float(tr)
        raise ValueError(f"No BIDS RepetitionTime metadata was found for {bold_path}.")

    def _build_contrast_map(self, plugin_cfg: dict[str, Any]) -> dict[str, str]:
        explicit = self._validate_contrast_config(plugin_cfg)
        return {str(name): str(value) for name, value in explicit.items()}

    def _infer_conditions(self, events: pd.DataFrame, plugin_cfg: dict[str, Any]) -> list[str]:
        configured = plugin_cfg.get("conditions")
        if isinstance(configured, list):
            return [str(value).strip() for value in configured if str(value).strip()]
        if "trial_type" not in events.columns:
            return []
        ordered = [
            str(value).strip()
            for value in events["trial_type"].dropna().unique()
            if str(value).strip()
        ]
        return list(dict.fromkeys(ordered))

    def _save_design_png(self, design_matrix: pd.DataFrame, path: Path) -> None:
        from nilearn.plotting import plot_design_matrix

        figure = plot_design_matrix(design_matrix, rescale=True)
        if hasattr(figure, "savefig"):
            figure.savefig(path, bbox_inches="tight")
            return
        raise TypeError("Nilearn plot_design_matrix did not return a figure-like object for the current version.")

    def _summarize_motion_qc(self, confounds: pd.DataFrame, fd_threshold: float) -> dict[str, float | int]:
        if confounds.empty or "framewise_displacement" not in confounds.columns:
            return {
                "mean_fd": 0.0,
                "max_fd": 0.0,
                "num_above_threshold": 0,
                "percent_above_threshold": 0.0,
                "threshold": float(fd_threshold),
            }

        fd_values = pd.to_numeric(confounds["framewise_displacement"], errors="coerce").fillna(0.0)
        above_threshold = (fd_values > fd_threshold).sum()
        total_values = len(fd_values)
        return {
            "mean_fd": float(fd_values.mean()),
            "max_fd": float(fd_values.max()),
            "num_above_threshold": int(above_threshold),
            "percent_above_threshold": float((above_threshold / total_values) * 100.0) if total_values else 0.0,
            "threshold": float(fd_threshold),
        }

    def _output_path(
        self,
        output_dir: str | Path,
        *,
        subject: str,
        session: str | None = None,
        task: str | None = None,
        run: str | None = None,
        desc: str | None = None,
        stat: str | None = None,
        suffix: str = ".nii.gz",
    ) -> Path:
        return DerivativePathBuilder.build(
            output_dir,
            subject=subject,
            session=session,
            task=task,
            run=run,
            desc=desc,
            stat=stat,
            suffix=suffix,
        )

