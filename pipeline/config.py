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


def load_default_config(root_dir="."):
    config = _apply_defaults({})
    config_dir = Path(root_dir)
    return _resolve_paths(config, config_dir)


def _apply_defaults(config):
    defaults = {
        "study_root": ".",
        "log_dir": "",
        "tokens": {
            "anat": ["mprage", "t1", "t2", "anat", "mpr", "sag", "t1w", "t2w"],
            "func": ["bold", "fmri", "rest", "nback", "flanker", "task", "functional", "sbref"],
            "fmap": ["fieldmap", "field_map", "fmap", "phase", "phasediff", "magnitude"],
            "dwi": ["dwi", "diff", "dtifit", "dti"]
        },
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

    user_tokens = config.get("tokens", {})
    merged_tokens = {}
    for token_type, default_list in defaults["tokens"].items():
        extra_list = user_tokens.get(token_type, [])
        merged_tokens[token_type] = list(dict.fromkeys(default_list + extra_list))
    merged["tokens"] = merged_tokens
    return merged


def _resolve_paths(config, root_dir):
    study_root = Path(config.get("study_root", "."))
    if not study_root.is_absolute():
        study_root = (root_dir / study_root).resolve()

    config["study_root"] = str(study_root)
    if not config.get("log_dir"):
        config["log_dir"] = str(Path(study_root) / "logs")
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
