"""Load and validate investigator-authored analysis decision manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DecisionManifest:
    """A validated decision manifest and its source metadata."""

    analysis_id: str
    path: Path
    content_hash: str
    data: dict[str, Any]

    @property
    def population(self) -> dict[str, Any]:
        population = self.data.get("population", {})
        return population if isinstance(population, dict) else {}


def load_decision_manifest(
    path: str | Path,
    *,
    expected_analysis_id: str | None = None,
) -> DecisionManifest:
    """Load one investigator-authored YAML decision manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Decision manifest not found: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Decision manifest is not a file: {manifest_path}")

    content = manifest_path.read_bytes()
    try:
        data = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not parse decision manifest {manifest_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Decision manifest {manifest_path} must contain a mapping.")

    analysis_id = str(data.get("analysis_id", "")).strip()
    if not analysis_id:
        raise ValueError(f"Decision manifest {manifest_path} must define analysis_id.")
    if expected_analysis_id is not None and analysis_id != expected_analysis_id:
        raise ValueError(
            f"Decision manifest {manifest_path} declares analysis_id '{analysis_id}', "
            f"but is registered as '{expected_analysis_id}'."
        )
    if data.get("schema_version") != 1:
        raise ValueError(f"Decision manifest {manifest_path} must use schema_version 1.")
    _validate_population(data.get("population", {}), manifest_path)

    content_hash = hashlib.sha256(content).hexdigest()
    print(
        f"[decisions] loading manifest '{analysis_id}' from {manifest_path} "
        f"(sha256={content_hash[:12]}...)"
    )
    return DecisionManifest(
        analysis_id=analysis_id,
        path=manifest_path,
        content_hash=content_hash,
        data=data,
    )


def load_configured_decisions(config: Mapping[str, Any]) -> dict[str, DecisionManifest]:
    """Load all manifests registered by the top-level config decisions mapping."""

    configured = config.get("decisions", {})
    if not isinstance(configured, Mapping):
        raise ValueError("The 'decisions' configuration must be an object.")
    return {
        str(analysis_id): load_decision_manifest(
            manifest_path,
            expected_analysis_id=str(analysis_id),
        )
        for analysis_id, manifest_path in configured.items()
    }


def _validate_population(value: Any, manifest_path: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Decision manifest {manifest_path}.population must be a mapping.")
    unknown = set(value) - {"include", "exclude"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Decision manifest population has unsupported fields: {names}.")
    for field_name in ("include", "exclude"):
        rules = value.get(field_name, [])
        if not isinstance(rules, list):
            raise ValueError(f"Decision manifest population.{field_name} must be a list of selectors.")
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(
                    f"Decision manifest population.{field_name}[{index}] must be a mapping."
                )
            subject = str(rule.get("subject", "")).strip()
            if not subject:
                raise ValueError(
                    f"Decision manifest population.{field_name}[{index}].subject must not be empty."
                )
            unknown = set(rule) - {"subject", "session", "run"}
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Decision manifest population.{field_name}[{index}] has unsupported fields: {names}."
                )
            for entity_name in ("session", "run"):
                if entity_name not in rule:
                    continue
                values = rule[entity_name] if isinstance(rule[entity_name], list) else [rule[entity_name]]
                if not values or not all(str(value).strip() for value in values):
                    raise ValueError(
                        f"Decision manifest population.{field_name}[{index}].{entity_name} must be a non-empty value or list."
                    )


__all__ = ["DecisionManifest", "load_configured_decisions", "load_decision_manifest"]
