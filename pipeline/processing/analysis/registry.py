from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Type

from .base import AnalysisPlugin, AnalysisResult
from .dataset import DatasetIndex

logger = logging.getLogger(__name__)


class AnalysisPluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, Type[AnalysisPlugin]] = {}

    def register(self, plugin_cls: Type[AnalysisPlugin]) -> Type[AnalysisPlugin]:
        if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, AnalysisPlugin):
            raise TypeError("Registered analysis plugins must subclass AnalysisPlugin.")

        plugin = plugin_cls()
        self._plugins[plugin.name] = plugin_cls
        return plugin_cls

    def get(self, name: str) -> Type[AnalysisPlugin] | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        return sorted(self._plugins)

    def discover(self) -> "AnalysisPluginRegistry":
        self._load_builtin_plugins()
        self._load_entry_point_plugins()
        return self

    def _load_builtin_plugins(self) -> None:
        try:
            from .activation.plugin import ActivationAnalysisPlugin

            self.register(ActivationAnalysisPlugin)
        except Exception as exc:
            logger.exception("Failed to load built-in analysis plugin 'activation'.")
            raise RuntimeError(
                "Failed to load built-in analysis plugin 'activation'. "
                "Verify that the installed package includes the pipeline processing analysis modules "
                "and that its dependencies are available."
            ) from exc

    def _load_entry_point_plugins(self) -> None:
        try:
            groups = entry_points()
            if hasattr(groups, "select"):
                discovered = groups.select(group="hbicproc.analysis")
            else:
                discovered = groups.get("hbicproc.analysis", [])
        except Exception:
            discovered = []

        for entry_point in discovered:
            try:
                plugin_cls = entry_point.load()
            except Exception:
                logger.exception(
                    "Failed to load analysis plugin entry point '%s'.",
                    getattr(entry_point, "name", entry_point),
                )
                continue
            if isinstance(plugin_cls, type):
                self.register(plugin_cls)

    def run_for_subject(
        self,
        subject: str,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        rerun: bool = False,
        plugin_name: str | None = None,
        dataset_index: DatasetIndex | None = None,
    ) -> list[AnalysisResult]:
        self.discover()

        names = [plugin_name] if plugin_name else self.list_plugins()
        results: list[AnalysisResult] = []
        for name in names:
            plugin_cls = self.get(name)
            if plugin_cls is None:
                continue
            plugin = plugin_cls()
            plugin_config = plugin.get_config(config)
            if dataset_index is None and (rerun or plugin_config.get("enabled", False)):
                dataset_index = DatasetIndex.from_config(config)
            result = plugin.execute(
                subject,
                config,
                dry_run=dry_run,
                rerun=rerun,
                plugin_config=plugin_config,
                dataset_index=dataset_index,
            )
            results.append(result)

        if not results:
            return [
                AnalysisResult(
                    success=False,
                    plugin="registry",
                    message="No analysis plugins were discovered. Add a plugin to the registry or install a hbicproc.analysis entry point.",
                )
            ]

        return results


_GLOBAL_REGISTRY: AnalysisPluginRegistry | None = None


def get_analysis_registry() -> AnalysisPluginRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = AnalysisPluginRegistry().discover()
    return _GLOBAL_REGISTRY
