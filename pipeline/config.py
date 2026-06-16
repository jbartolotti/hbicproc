import json
from pathlib import Path


def load_config(path="pipeline_config.json"):
    config_path = Path(path)
    if not config_path.exists():
        candidate = Path.cwd() / "code" / config_path.name
        if candidate.exists():
            config_path = candidate
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config_dir = config_path.parent
    config = _apply_defaults(config)
    config = _resolve_paths(config, config_dir)
    return config


def save_default_config(path="pipeline_config.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = _apply_defaults({})
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


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
            "server": "https://xnat.example.org",
            "project_id": "MY_PROJECT",
            "credentials_file": "pipeline/xnat_credentials.example.json",
            "session_names_file": "",
            "session_names_delimiter": "_",
            "output_dir": "sourcedata",
            "verify_ssl": True
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
            "fmriprep_image": "docker://poldracklab/fmriprep:latest",
            "fastsurfer_image": "",
            "freesurfer_image": "",
            "output_dir": "derivatives/fmriprep",
            "work_dir": "work/fmriprep",
            "freesurfer_subjects_dir": "derivatives/freesurfer",
            "fs_license_file": "/path/to/license.txt",
            "extra_args": "participant --participant_label {subject}",
            "use_slurm": False,
            "freesurfer_mode": "integrated",
            "slurm": {
                "partition": "sixhour",
                "cpus_per_task": 8,
                "mem": "32G",
                "time_fmriprep": "06:00:00",
                "time_fastsurfer": "02:00:00",
                "gpu_type": "a100"
            }
        },
        "hbicproc": {
            "exclusions_file": "derivatives/hbicproc/exclusions.json"
        }
    }

    merged = defaults.copy()
    merged.update(config)

    for key in ["xnat", "bidskit", "mriqc", "fmriprep", "hbicproc"]:
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

    if config.get("bids_root"):
        config["bids_root"] = str(_resolve_path(config["bids_root"], root_dir, study_root))

    for section in ["xnat", "bidskit", "mriqc", "fmriprep", "hbicproc"]:
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
