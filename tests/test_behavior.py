from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from pipeline.processing.analysis.derivatives import DerivativePathBuilder
from pipeline.processing.behavior.config import TaskConfig
from pipeline.processing.behavior.models import EventTable
from pipeline.processing.behavior.parsers.registry import ParserRegistry
from pipeline.processing.behavior.parsers.stroop import StroopParser
from pipeline.processing.behavior.validator import validate_event_table
from pipeline.processing.behavior.writer import EventsWriter


def test_task_config_loads_json(tmp_path: Path) -> None:
    config_path = tmp_path / "stroop.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "stroop",
                "trigger_column": "ScannerTriggerTime",
                "event_rows": {
                    "filter_column": "TrialType",
                    "include_values": ["Congruent", "Incongruent"],
                },
                "columns": {
                    "onset": {"source": "StimulusOnsetTime"},
                    "duration": {"source": "StimulusDuration"},
                    "trial_type": {"source": "TrialType"},
                },
                "optional_columns": ["Accuracy", "RT"],
            }
        ),
        encoding="utf-8",
    )

    config = TaskConfig.load(config_path)

    assert config.task == "stroop"
    assert config.trigger_column == "ScannerTriggerTime"
    assert config.columns["onset"].source == "StimulusOnsetTime"
    assert config.optional_columns == ["Accuracy", "RT"]


def test_stroop_parser_builds_events_dataframe() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "TrialType": "Congruent",
                "StimulusOnsetTime": 2.5,
                "StimulusDuration": 1.0,
                "Accuracy": 1,
                "RT": 0.45,
                "ScannerTriggerTime": 0.0,
            },
            {
                "TrialType": "Incongruent",
                "StimulusOnsetTime": 4.0,
                "StimulusDuration": 1.0,
                "Accuracy": 0,
                "RT": 0.62,
                "ScannerTriggerTime": 0.0,
            },
        ]
    )

    config = TaskConfig(
        task="stroop",
        trigger_column="ScannerTriggerTime",
        event_rows={"filter_column": "TrialType", "include_values": ["Congruent", "Incongruent"]},
        columns={
            "onset": {"source": "StimulusOnsetTime"},
            "duration": {"source": "StimulusDuration"},
            "trial_type": {"source": "TrialType"},
        },
        optional_columns=["Accuracy", "RT"],
    )

    parser = StroopParser()
    events_df = parser.parse(raw_df, config)

    assert list(events_df.columns)[:3] == ["onset", "duration", "trial_type"]
    assert events_df.loc[0, "onset"] == 2.5
    assert events_df.loc[0, "trial_type"] == "congruent"
    assert events_df.loc[0, "accuracy"] == 1
    assert events_df.loc[1, "rt"] == 0.62


def test_writer_writes_tsv_and_json_sidecar(tmp_path: Path) -> None:
    events_df = pd.DataFrame(
        [
            {"onset": 2.34, "duration": 1.0, "trial_type": "congruent", "rt": 0.45, "accuracy": 1},
        ]
    )
    output_path = tmp_path / "sub-001" / "ses-01"
    output_path.mkdir(parents=True, exist_ok=True)

    writer = EventsWriter()
    writer.write(events_df, output_path / "task-stroop_events.tsv", sidecar_path=output_path / "task-stroop_events.json")

    assert (output_path / "task-stroop_events.tsv").exists()
    assert (output_path / "task-stroop_events.json").exists()
    written = pd.read_csv(output_path / "task-stroop_events.tsv", sep="\t")
    assert list(written.columns) == ["onset", "duration", "trial_type", "rt", "accuracy"]


def test_registry_and_validator() -> None:
    registry = ParserRegistry()
    registry.register("stroop", StroopParser)

    parser_cls = registry.get("stroop")
    assert parser_cls is StroopParser

    table = EventTable.from_records([
        {"onset": 1.0, "duration": 1.0, "trial_type": "congruent"},
    ])
    validate_event_table(table)


def test_derivative_builder_normalizes_bids_entities() -> None:
    path = DerivativePathBuilder.build_file(
        Path("derivatives"),
        subject="sub_001",
        session="ses_baseline",
        task="task_rest",
        run="run_2",
        namespace="model",
        namespace_name="canonical_glm",
        desc="condition_a",
        stat="effect",
        suffix=".nii.gz",
    )

    expected = (
        Path("derivatives")
        / "sub-001"
        / "ses-baseline"
        / "func"
        / "task-rest"
        / "run-2"
        / "models"
        / "canonical_glm"
        / "condition_a_effect.nii.gz"
    )
    assert path == expected
