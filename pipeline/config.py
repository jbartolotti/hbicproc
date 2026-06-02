import json
import os
from pathlib import Path


def load_config(path="pipeline_config.json"):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config_dir = config_path.parent
    config = _apply_defaults(config)
    config = _resolve_paths(config, config_dir)
    return config


def _apply_defaults(config):
    defaults = {
        "study_root": ".",
        "log_dir": "logs",
        "xnat": {
            "script_path": "scripts/xnat_download.R",
            "output_dir": "sourcedata"
        },
        "bidskit": {
            "input_dir": "sourcedata",
            "output_dir": "work/bidskit",
            "extra_args": ""
        },
        "mriqc": {
            "singularity_image": "docker://poldracklab/mriqc:latest",
            "output_dir": "derivatives/mriqc",
            "work_dir": "work/mriqc",
            "extra_args": "participant --participant_label {subject}"
        },
        "fmriprep": {
            "singularity_image": "docker://poldracklab/fmriprep:latest",
            "output_dir": "derivatives/fmriprep",
            "work_dir": "work/fmriprep",
            "fs_license_file": "/path/to/license.txt",
            "extra_args": "participant --participant_label {subject}"
        }
    }

    merged = defaults.copy()
    merged.update(config)

    for key in ["xnat", "bidskit", "mriqc", "fmriprep"]:
        merged[key] = {**defaults.get(key, {}), **config.get(key, {})}

    return merged


def _resolve_paths(config, root_dir):
    study_root = Path(config.get("study_root", "."))
    if not study_root.is_absolute():
        study_root = (root_dir / study_root).resolve()

    config["study_root"] = str(study_root)
    config["log_dir"] = str(_resolve_path(config["log_dir"], root_dir, study_root))

    for section in ["xnat", "bidskit", "mriqc", "fmriprep"]:
        section_data = config.get(section, {})
        for key, value in section_data.items():
            if key.endswith("_dir") or key.endswith("_path") or key.endswith("_file"):
                section_data[key] = str(_resolve_path(value, root_dir, study_root))
        config[section] = section_data

    return config


def _resolve_path(value, config_dir, study_root):
    if not value:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    local = config_dir / value
    if local.exists() or not study_root.exists():
        return str(local.resolve())
    return str((study_root / value).resolve())
