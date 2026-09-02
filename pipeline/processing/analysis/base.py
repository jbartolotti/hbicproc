from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .dataset import DatasetIndex


@dataclass
class AnalysisResult:
    success: bool
    message: str
    plugin: str = ""
    skipped: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class AnalysisPlugin(ABC):
    name = "base"
    description = "Base analysis plugin."

    def get_config(self, config: dict[str, Any]) -> dict[str, Any]:
        analysis_config = config.get("analysis", {})
        subject_config = analysis_config.get("subject", {})
        plugin_config = subject_config.get(self.name, {})
        if not isinstance(plugin_config, dict):
            return {"enabled": False}

        merged = {"enabled": False}
        merged.update(plugin_config)
        return merged

    def execute(
        self,
        subject: str,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        rerun: bool = False,
        plugin_config: dict[str, Any] | None = None,
        dataset_index: DatasetIndex | None = None,
    ) -> AnalysisResult:
        plugin_cfg = plugin_config or self.get_config(config)
        enabled = plugin_cfg.get("enabled", False)
        if not enabled and not rerun:
            return AnalysisResult(
                success=True,
                skipped=True,
                plugin=self.name,
                message=f"Analysis plugin '{self.name}' is disabled for subject {subject}.",
                details={"enabled": False},
            )

        return self.run(
            subject,
            config,
            dry_run=dry_run,
            rerun=rerun,
            plugin_config=plugin_cfg,
            dataset_index=dataset_index,
        )

    @abstractmethod
    def run(
        self,
        subject: str,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        rerun: bool = False,
        plugin_config: dict[str, Any] | None = None,
        dataset_index: DatasetIndex | None = None,
    ) -> AnalysisResult:
        raise NotImplementedError("Analysis plugin implementations must override run().")
