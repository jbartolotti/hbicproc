from __future__ import annotations

import logging
from typing import Any

from .dataset import DatasetIndex
from .registry import get_analysis_registry

logger = logging.getLogger(__name__)


def _enabled_plugin_names(config: dict[str, Any]) -> list[str]:
    analysis_config = config.get("analysis", {})
    subject_config = analysis_config.get("subject", {})
    if not isinstance(subject_config, dict):
        return []

    enabled: list[str] = []
    for name, plugin_cfg in subject_config.items():
        if isinstance(plugin_cfg, dict) and plugin_cfg.get("enabled"):
            enabled.append(name)
    return enabled


def run(
    subject: str,
    config: dict[str, Any],
    dry_run: bool = False,
    rerun: bool = False,
    plugin_name: str | None = None,
    dataset_index: DatasetIndex | None = None,
) -> dict[str, Any]:
    try:
        registry = get_analysis_registry()
    except Exception as exc:
        logger.exception("Analysis plugin discovery failed for subject=%s.", subject)
        return {
            "success": False,
            "skipped": False,
            "message": f"Analysis plugin loading failed for subject {subject}: {exc}",
            "details": {"subject": subject, "error": str(exc)},
        }
    if plugin_name:
        requested = [plugin_name]
    else:
        requested = _enabled_plugin_names(config) or registry.list_plugins()

    if plugin_name and registry.get(plugin_name) is None:
        return {
            "success": False,
            "skipped": False,
            "message": f"Unknown analysis plugin: {plugin_name}",
            "details": {"subject": subject, "plugins": requested},
        }

    results = []
    for name in requested:
        plugin_cls = registry.get(name)
        if plugin_cls is None:
            continue
        plugin = plugin_cls()
        plugin_config = plugin.get_config(config)
        if dataset_index is None and (rerun or plugin_config.get("enabled", False)):
            dataset_index = DatasetIndex.from_config(config)
        results.append(
            plugin.execute(
                subject,
                config,
                dry_run=dry_run,
                rerun=rerun,
                plugin_config=plugin_config,
                dataset_index=dataset_index,
            )
        )

    if not results:
        return {
            "success": False,
            "skipped": False,
            "message": (
                f"No analysis plugins were available for subject {subject}. "
                "Plugin discovery completed without registering any plugins."
            ),
            "details": {"subject": subject, "plugins": requested},
        }

    skipped = all(result.skipped for result in results)
    success = all(result.success for result in results)
    details = {
        "subject": subject,
        "plugins": requested,
        "results": [result.__dict__ for result in results],
    }

    return {
        "success": success,
        "skipped": skipped,
        "message": "\n".join(result.message for result in results),
        "details": details,
    }
