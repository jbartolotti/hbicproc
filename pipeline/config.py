from pathlib import Path

import yaml


def load_config(path="pipeline_config.yaml"):
    config_path = Path(path)
    if not config_path.exists():
        candidate = Path.cwd() / "code" / config_path.name
        if candidate.exists():
            config_path = candidate
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        config = {}

    config_dir = config_path.parent
    config = _apply_defaults(config)
    validate_config(config)
    config = _resolve_paths(config, config_dir)
    return config


def save_default_config(path="pipeline_config.yaml"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = _apply_defaults({})
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def load_default_config(root_dir="."):
    config = _apply_defaults({})
    validate_config(config)
    config_dir = Path(root_dir)
    return _resolve_paths(config, config_dir)


def _apply_defaults(config):
    defaults = {
        "study_root": ".",
        "code_dir": "code",
        "email": "",
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
            "output_dir": "sourcedata",
            "session_names_delimiter": "_",
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
        },
        "analysis": {
            "output_dir": "derivatives/hbicproc",
            "input_dataset": {
                "name": "fmriprep",
                "path": "derivatives/fmriprep"
            },
            "subject": {
                "tasks": {}
            }
        },
        "behavior": {
            "enabled": False,
            "tasks": [],
            "task_configs_dir": "code/behavior",
            "parser_plugins_dir": "code/behavior/plugins",
            "edat3_search_root": "sourcedata",
            "output_dir": "derivatives/behavior/events",
            "reader_backend": "auto",
            "verbose": False
        }
    }

    merged = _deep_merge(defaults, config)

    for key in ["xnat", "bidskit", "mriqc", "fmriprep", "hbicproc", "analysis", "behavior"]:
        merged[key] = _deep_merge(defaults.get(key, {}), config.get(key, {}))

    user_tokens = config.get("tokens", {})
    merged_tokens = {}
    for token_type, default_list in defaults["tokens"].items():
        extra_list = user_tokens.get(token_type, [])
        merged_tokens[token_type] = list(dict.fromkeys(default_list + extra_list))
    merged["tokens"] = merged_tokens
    _normalize_analysis_configuration(merged["analysis"])
    return merged


def _normalize_analysis_configuration(analysis):
    if not isinstance(analysis, dict):
        return

    subject = analysis.setdefault("subject", {})
    if not isinstance(subject, dict):
        return
    tasks = subject.setdefault("tasks", {})
    if not isinstance(tasks, dict):
        return

    for task_config in tasks.values():
        if isinstance(task_config, dict):
            task_config.setdefault("enabled", True)
            task_config.setdefault("models", {})
            task_config.setdefault("analyses", {})
            analyses = task_config.get("analyses")
            if isinstance(analyses, dict):
                for analysis_config in analyses.values():
                    if isinstance(analysis_config, dict):
                        analysis_config.setdefault("enabled", True)


def validate_config(config):
    """Validate the authoritative JSON configuration schema."""

    analysis = config.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("The 'analysis' configuration must be an object.")
    if not str(analysis.get("output_dir", "")).strip():
        raise ValueError("analysis.output_dir must not be empty.")

    input_dataset = analysis.get("input_dataset")
    _validate_input_dataset(input_dataset, field_name="analysis.input_dataset")

    subject = analysis.get("subject")
    if not isinstance(subject, dict):
        raise ValueError("analysis.subject must be an object.")
    if "activation" in subject:
        raise ValueError(
            "analysis.subject.activation is obsolete; configure analyses under analysis.subject.tasks."
        )

    tasks = subject.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("analysis.subject.tasks must be an object keyed by task name.")

    for task_name, task_config in tasks.items():
        if not str(task_name).strip():
            raise ValueError("Analysis task names must not be empty.")
        if not isinstance(task_config, dict):
            raise ValueError(f"analysis.subject.tasks.{task_name} must be an object.")
        if not isinstance(task_config.get("enabled", True), bool):
            raise ValueError(f"analysis.subject.tasks.{task_name}.enabled must be a boolean.")
        if "input_dataset" in task_config:
            _validate_input_dataset(
                task_config["input_dataset"],
                field_name=f"analysis.subject.tasks.{task_name}.input_dataset",
            )

        models = task_config.get("models", {})
        if not isinstance(models, dict):
            raise ValueError(f"analysis.subject.tasks.{task_name}.models must be an object.")
        for model_name, model_config in models.items():
            if not str(model_name).strip() or not isinstance(model_config, dict):
                raise ValueError(
                    f"analysis.subject.tasks.{task_name}.models entries must be named objects."
                )
            if not str(model_config.get("type", "")).strip():
                raise ValueError(
                    f"analysis.subject.tasks.{task_name}.models.{model_name}.type must not be empty."
                )

        analyses = task_config.get("analyses", {})
        if not isinstance(analyses, dict):
            raise ValueError(f"analysis.subject.tasks.{task_name}.analyses must be an object.")
        for analysis_name, analysis_config in analyses.items():
            field_name = f"analysis.subject.tasks.{task_name}.analyses.{analysis_name}"
            if not str(analysis_name).strip() or not isinstance(analysis_config, dict):
                raise ValueError(f"{field_name} must be a named object.")
            if not isinstance(analysis_config.get("enabled", True), bool):
                raise ValueError(f"{field_name}.enabled must be a boolean.")
            model_name = str(analysis_config.get("model", "")).strip()
            if not model_name:
                raise ValueError(f"{field_name}.model must reference a model.")
            if model_name not in models:
                raise ValueError(f"{field_name}.model references unknown model '{model_name}'.")


def _validate_input_dataset(value, *, field_name):
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object with 'name' and 'path'.")
    if not str(value.get("name", "")).strip():
        raise ValueError(f"{field_name}.name must not be empty.")
    if not str(value.get("path", "")).strip():
        raise ValueError(f"{field_name}.path must not be empty.")


def _deep_merge(base, override):
    merged = {}
    for key, value in base.items():
        if isinstance(value, dict) and isinstance(override.get(key), dict):
            merged[key] = _deep_merge(value, override[key])
        else:
            merged[key] = override.get(key, value)
    for key, value in override.items():
        if key not in merged:
            merged[key] = value
    return merged


def _resolve_paths(config, root_dir):
    study_root = Path(config.get("study_root", "."))
    if not study_root.is_absolute():
        study_root = (root_dir / study_root).resolve()

    config["study_root"] = str(study_root)
    if not config.get("log_dir"):
        config["log_dir"] = str(Path(study_root) / "logs")
    config["log_dir"] = str(_resolve_path(config["log_dir"], root_dir, study_root))

    if not config.get("code_dir"):
        config["code_dir"] = str(Path(study_root) / "code")
    config["code_dir"] = str(_resolve_path(config["code_dir"], root_dir, study_root))

    if config.get("bids_root"):
        config["bids_root"] = str(_resolve_path(config["bids_root"], root_dir, study_root))

    for section in ["xnat", "bidskit", "mriqc", "fmriprep", "hbicproc", "analysis", "behavior"]:
        section_data = config.get(section, {})
        _resolve_nested_paths(section_data, root_dir, study_root)
        config[section] = section_data

    _resolve_analysis_input_dataset_paths(config.get("analysis", {}), root_dir, study_root)

    return config


def _resolve_analysis_input_dataset_paths(analysis, config_dir, study_root):
    if not isinstance(analysis, dict):
        return

    input_dataset = analysis.get("input_dataset")
    if isinstance(input_dataset, dict) and isinstance(input_dataset.get("path"), str):
        input_dataset["path"] = str(_resolve_path(input_dataset["path"], config_dir, study_root))

    subject = analysis.get("subject")
    tasks = subject.get("tasks") if isinstance(subject, dict) else None
    if not isinstance(tasks, dict):
        return
    for task_config in tasks.values():
        if not isinstance(task_config, dict):
            continue
        task_input_dataset = task_config.get("input_dataset")
        if isinstance(task_input_dataset, dict) and isinstance(task_input_dataset.get("path"), str):
            task_input_dataset["path"] = str(
                _resolve_path(task_input_dataset["path"], config_dir, study_root)
            )


def _resolve_nested_paths(section_data, root_dir, study_root):
    for key, value in section_data.items():
        if isinstance(value, dict):
            _resolve_nested_paths(value, root_dir, study_root)
        elif isinstance(value, str) and (key.endswith("_dir") or key.endswith("_path") or key.endswith("_file")):
            section_data[key] = str(_resolve_path(value, root_dir, study_root))


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
