from pathlib import Path

import pytest
import pandas as pd
import yaml

from pipeline.config import _apply_defaults, _resolve_paths, validate_config
from pipeline.decisions import load_configured_decisions, load_decision_manifest
from pipeline.processing.group.population import select_records
from pipeline.processing.group import service as group_service


def test_config_resolves_registered_decision_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "code" / "decisions" / "primary.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "schema_version: 1\nanalysis_id: primary\n",
        encoding="utf-8",
    )
    config = _apply_defaults({"decisions": {"primary": "code/decisions/primary.yaml"}})
    validate_config(config)
    resolved = _resolve_paths(config, tmp_path)

    decisions = load_configured_decisions(resolved)

    assert decisions["primary"].path == manifest_path
    assert len(decisions["primary"].content_hash) == 64


def test_decision_manifest_requires_matching_registered_analysis_id(tmp_path: Path) -> None:
    manifest_path = tmp_path / "wrong.yaml"
    manifest_path.write_text("schema_version: 1\nanalysis_id: other\n", encoding="utf-8")

    with pytest.raises(ValueError, match="registered as 'primary'"):
        load_decision_manifest(manifest_path, expected_analysis_id="primary")


def test_decision_manifest_requires_supported_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "unsupported.yaml"
    manifest_path.write_text("schema_version: 2\nanalysis_id: primary\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version 1"):
        load_decision_manifest(manifest_path)


def test_population_rules_select_subject_sessions_and_runs() -> None:
    numeric_subject = yaml.safe_load("subject: 001")["subject"]
    records = pd.DataFrame(
        [
            {"subject": "001", "session": "BL", "run": "1"},
            {"subject": "001", "session": "BL", "run": "2"},
            {"subject": "001", "session": "w12", "run": "1"},
            {"subject": "002", "session": "week1", "run": "1"},
            {"subject": "002", "session": "week2", "run": "1"},
            {"subject": "002", "session": "week3", "run": "1"},
        ]
    )

    selected = select_records(
        records,
        {
            "exclude": [
                {"subject": numeric_subject, "session": ["ses-BL", "w12"]},
                {"subject": "sub-002", "session": ["ses-week1", "ses-week3"]},
            ],
        },
    )

    assert selected[["subject", "session", "run"]].to_dict("records") == [
        {"subject": "002", "session": "week2", "run": "1"},
    ]

    selected_run = select_records(
        records,
        {
            "include": [{"subject": numeric_subject, "session": "ses-BL"}],
            "exclude": [{"subject": "sub-001", "session": "BL", "run": 2}],
        },
    )
    assert selected_run[["subject", "session", "run"]].to_dict("records") == [
        {"subject": "001", "session": "BL", "run": "1"}
    ]


def test_group_one_sample_resolves_registered_decision(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "primary.yaml"
    manifest_path.write_text(
        "schema_version: 1\n"
        "analysis_id: primary\n"
        "population:\n"
        "  include:\n"
        "    - subject: sub-001\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run_one_sample(config, specification, *, decision_manifest=None):
        captured["manifest"] = decision_manifest
        return {"analysis": "one_sample"}

    monkeypatch.setattr(group_service, "run_one_sample", fake_run_one_sample)
    config = _apply_defaults(
        {
            "decisions": {"primary": str(manifest_path)},
            "group": {
                "one_sample": {
                    "enabled": True,
                    "decision": "primary",
                    "contrasts": ["memory"],
                    "inference": {"method": "fdr", "alpha": 0.05},
                }
            },
        }
    )
    validate_config(config)
    resolved = _resolve_paths(config, tmp_path)

    result = group_service.run(resolved)

    assert result["success"] is True
    assert captured["manifest"].analysis_id == "primary"
    assert captured["manifest"].population["include"][0]["subject"] == "sub-001"


def test_decision_manifest_rejects_legacy_parallel_population_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "legacy.yaml"
    manifest_path.write_text(
        "schema_version: 1\n"
        "analysis_id: legacy\n"
        "population:\n"
        "  include_subjects: [sub-001]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields: include_subjects"):
        load_decision_manifest(manifest_path)
