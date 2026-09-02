from .base import BaseStage, StageResult
from ..processing.analysis.dataset import DatasetIndex
from ..processing.analysis.service import run as analysis_run


class AnalysisStage(BaseStage):
    name = "analysis"
    # Analysis completion is determined per run from its core output products.
    state_key = None

    def __init__(self):
        self._dataset_index = None

    def _get_dataset_index(self, config):
        if self._dataset_index is None:
            self._dataset_index = DatasetIndex.from_config(config)
        return self._dataset_index

    def _requires_dataset_index(self, config, *, rerun=False, plugin_name=None):
        if rerun:
            return True

        analysis_config = config.get("analysis", {})
        subject_config = analysis_config.get("subject", {}) if isinstance(analysis_config, dict) else {}
        if not isinstance(subject_config, dict):
            return False

        if plugin_name:
            plugin_config = subject_config.get(plugin_name, {})
            return isinstance(plugin_config, dict) and bool(plugin_config.get("enabled", False))

        return any(
            isinstance(plugin_config, dict) and plugin_config.get("enabled", False)
            for plugin_config in subject_config.values()
        )

    def run(self, subject, config, state, dry_run=False, rerun=False, plugin_name=None, dataset_index=None):
        if self._requires_dataset_index(config, rerun=rerun, plugin_name=plugin_name):
            if dataset_index is None:
                dataset_index = self._get_dataset_index(config)

        result = analysis_run(
            subject,
            config,
            dry_run=dry_run,
            rerun=rerun,
            plugin_name=plugin_name,
            dataset_index=dataset_index,
        )

        return StageResult(
            success=result.get("success", False),
            skipped=result.get("skipped", False),
            message=result.get("message", ""),
            next_command=result.get("next_command"),
            details=result.get("details", {}),
        )
