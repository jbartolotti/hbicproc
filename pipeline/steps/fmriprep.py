import shlex
import subprocess
from pathlib import Path

from ._utils import path_exists, run_subprocess
from ..utils import get_bids_root, ensure_dir


def get_fmriprep_subjects(config):
    bids_root = get_bids_root(config)
    if not bids_root.exists():
        return []
    return sorted([p.name for p in bids_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])


def is_fmriprep_complete(subject, config):
    output_dir = Path(config["fmriprep"]["output_dir"]) / subject
    report_file = output_dir / "html" / f"{subject}.html"
    return output_dir.is_dir() and report_file.exists()


def _resolve_fmriprep_image(config):
    return config["fmriprep"].get("fmriprep_image") or config["fmriprep"].get("singularity_image")


def _resolve_freesurfer_subjects_dir(config):
    fs_dir = config["fmriprep"].get("freesurfer_subjects_dir")
    if fs_dir:
        return Path(fs_dir)
    return Path(config["study_root"]) / "derivatives" / "freesurfer"


def _find_t1w_file(subject, config):
    bids_root = get_bids_root(config)
    subject_dir = bids_root / subject
    if not subject_dir.exists():
        return None

    for path in sorted(subject_dir.rglob("*_T1w.nii*")):
        if path.is_file():
            return path
    return None


def _find_multi_echo(subject, config):
    bids_root = get_bids_root(config)
    subject_dir = bids_root / subject
    if not subject_dir.exists():
        return False
    return any(subject_dir.rglob("*echo-*.nii*"))


def _has_cuda():
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _get_fmriprep_cpus(config):
    slurm = config["fmriprep"].get("slurm", {})
    return int(slurm.get("cpus_per_task", 8))


def run_recon_all(subject, config, dry_run=False):
    t1w_file = _find_t1w_file(subject, config)
    if not t1w_file:
        return {
            "success": False,
            "skipped": False,
            "message": f"Unable to find T1w file for {subject} in BIDS dataset.",
            "command": "",
            "returncode": None,
        }

    freesurfer_dir = _resolve_freesurfer_subjects_dir(config)
    ensure_dir(freesurfer_dir)

    subject_dir = freesurfer_dir / subject
    if subject_dir.exists():
        return {
            "success": True,
            "skipped": True,
            "message": f"Freesurfer output already exists at {subject_dir}",
            "command": "",
            "returncode": None,
        }

    freesurfer_image = config["fmriprep"].get("freesurfer_image")
    if freesurfer_image:
        command = [
            "singularity",
            "exec",
            freesurfer_image,
            "recon-all",
            "-s",
            subject,
            "-i",
            str(t1w_file),
            "-sd",
            str(freesurfer_dir),
            "-all",
        ]
    else:
        command = [
            "recon-all",
            "-s",
            subject,
            "-i",
            str(t1w_file),
            "-sd",
            str(freesurfer_dir),
            "-all",
        ]

    return run_subprocess(command, dry_run=dry_run)


def run_fastsurfer(subject, config, dry_run=False):
    t1w_file = _find_t1w_file(subject, config)
    if not t1w_file:
        return {
            "success": False,
            "skipped": False,
            "message": f"Unable to find T1w file for {subject} in BIDS dataset.",
            "command": "",
            "returncode": None,
        }

    freesurfer_dir = _resolve_freesurfer_subjects_dir(config)
    ensure_dir(freesurfer_dir)

    fastsurfer_image = config["fmriprep"].get("fastsurfer_image")
    if not fastsurfer_image:
        return {
            "success": False,
            "skipped": False,
            "message": "FastSurfer image is not configured. Set fmriprep.fastsurfer_image in your config.",
            "command": "",
            "returncode": None,
        }

    command = ["singularity", "exec"]
    if _has_cuda():
        command.append("--nv")
    command.extend([
        fastsurfer_image,
        "fastsurfer.sh",
        "--t1",
        str(t1w_file),
        "--sid",
        subject,
        "--sd",
        str(freesurfer_dir),
        "--threads",
        str(_get_fmriprep_cpus(config)),
    ])

    return run_subprocess(command, dry_run=dry_run)


def _build_fmriprep_command(subject, config):
    bids_root = get_bids_root(config)
    if not bids_root.exists():
        return None, {
            "success": False,
            "skipped": False,
            "message": f"BIDS directory not found: {bids_root}",
            "command": "",
            "returncode": None,
        }

    fmriprep_image = _resolve_fmriprep_image(config)
    if not fmriprep_image:
        return None, {
            "success": False,
            "skipped": False,
            "message": "fMRIPrep singularity image is not configured.",
            "command": "",
            "returncode": None,
        }

    output_dir = Path(config["fmriprep"]["output_dir"]) / subject
    work_dir = Path(config["fmriprep"]["work_dir"]) / subject
    ensure_dir(output_dir)
    ensure_dir(work_dir)

    mode = config["fmriprep"].get("freesurfer_mode", "integrated")
    license_file = Path(config["fmriprep"].get("fs_license_file", ""))
    if mode != "none" and not license_file.exists():
        return None, {
            "success": False,
            "skipped": False,
            "message": f"FreeSurfer license file not found: {license_file}.",
            "command": "",
            "returncode": None,
        }

    extra_args = config["fmriprep"].get("extra_args", "").format(subject=subject).strip()
    command = ["singularity", "exec", "--cleanenv", fmriprep_image, "fmriprep", str(bids_root), str(output_dir)]

    if extra_args:
        extra_lower = extra_args.lower()
        if "--participant-label" not in extra_lower and "--participant_label" not in extra_lower:
            command.extend(["--participant-label", subject])
        command.extend(shlex.split(extra_args))
    else:
        command.extend(["--participant-label", subject])

    if mode == "none":
        command.append("--fs-no-reconall")
    elif mode in ("reuse", "freesurfer", "fastsurfer"):
        freesurfer_dir = _resolve_freesurfer_subjects_dir(config)
        if not freesurfer_dir.exists():
            return None, {
                "success": False,
                "skipped": False,
                "message": f"Freesurfer subjects directory not found: {freesurfer_dir}",
                "command": "",
                "returncode": None,
            }
        command.extend([
            "--fs-subjects-dir",
            str(freesurfer_dir),
            "--fs-no-reconall",
            "--fs-no-resume",
        ])
    elif mode == "integrated":
        if license_file.exists() and "--fs-license-file" not in extra_args:
            command.extend(["--fs-license-file", str(license_file)])

    if _find_multi_echo(subject, config):
        command.append("--me-output-echoes")

    if "--fs-license-file" not in extra_args and mode != "integrated" and license_file.exists():
        command.extend(["--fs-license-file", str(license_file)])

    return command, None


def run_fmriprep(subject, config, dry_run=False, rerun=False):
    output_dir = Path(config["fmriprep"]["output_dir"]) / subject
    if path_exists(output_dir) and not rerun and is_fmriprep_complete(subject, config):
        return {
            "success": True,
            "skipped": True,
            "message": f"fMRIPrep output already exists for {subject} at {output_dir}",
            "command": "",
            "returncode": None,
        }

    command, error_result = _build_fmriprep_command(subject, config)
    if error_result is not None:
        return error_result

    return run_subprocess(command, dry_run=dry_run)


def run_fmriprep_subject(subject, config, dry_run=False, rerun=False):
    if config["fmriprep"].get("use_slurm"):
        scripts = generate_slurm_scripts([subject], config)
        return {
            "success": True,
            "skipped": False,
            "message": f"SLURM mode enabled. Generated scripts: {', '.join(scripts)}",
            "command": "",
            "returncode": None,
        }

    mode = config["fmriprep"].get("freesurfer_mode", "integrated")

    if mode == "freesurfer":
        status = run_recon_all(subject, config, dry_run=dry_run)
        if not status["success"]:
            return status
    elif mode == "fastsurfer":
        status = run_fastsurfer(subject, config, dry_run=dry_run)
        if not status["success"]:
            return status

    return run_fmriprep(subject, config, dry_run=dry_run, rerun=rerun)


def run_fmriprep_batch(subjects, config, dry_run=False, rerun=False):
    if config["fmriprep"].get("use_slurm"):
        return generate_slurm_scripts(subjects, config)

    results = []
    for subject in subjects:
        results.append(run_fmriprep_subject(subject, config, dry_run=dry_run, rerun=rerun))
    return results


def run(subject, config, dry_run=False):
    return run_fmriprep_subject(subject, config, dry_run=dry_run)


def _slurm_header(name, slurm, array_length, output_path):
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={slurm.get('partition', 'sixhour')}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 8)}",
        f"#SBATCH --mem={slurm.get('mem', '32G')}",
        f"#SBATCH --time={slurm.get('time_fmriprep', '06:00:00') if name == 'slurm_fmriprep' else slurm.get('time_fastsurfer', '02:00:00')}",
        f"#SBATCH --array=0-{max(array_length - 1, 0)}",
        f"#SBATCH --output={output_path}",
        "set -euo pipefail",
        "",
    ]
    return "\n".join(lines)


def _quote(path):
    return shlex.quote(str(path))


def generate_slurm_scripts(subjects, config):
    work_dir = Path(config["fmriprep"]["work_dir"])
    ensure_dir(work_dir)

    subject_list_file = work_dir / "slurm_fmriprep_subjects.txt"
    subject_list_file.write_text("\n".join(subjects) + "\n")

    slurm = config["fmriprep"].get("slurm", {})
    mode = config["fmriprep"].get("freesurfer_mode", "integrated")
    scripts = []

    fs_dir = _resolve_freesurfer_subjects_dir(config)
    bids_root = get_bids_root(config)
    fmriprep_image = _resolve_fmriprep_image(config)
    fastsurfer_image = config["fmriprep"].get("fastsurfer_image")
    license_file = config["fmriprep"].get("fs_license_file", "")
    extra_args = config["fmriprep"].get("extra_args", "").format(subject="$SUBJECT")

    array_size = len(subjects)
    pre_job = None

    if mode in ("fastsurfer", "freesurfer"):
        pre_job = work_dir / "slurm_fmriprep_precompute.sh"
        pre_output = work_dir / "slurm_fmriprep_precompute_%A_%a.out"
        header = _slurm_header("slurm_fmriprep_precompute", slurm, array_size, _quote(pre_output))
        lines = [header]
        lines.append(f"SUBJECT=$(sed -n '$((SLURM_ARRAY_TASK_ID + 1))p' { _quote(subject_list_file) })")
        lines.append("if [[ -z \"$SUBJECT\" ]]; then echo 'Subject not found'; exit 1; fi")
        lines.append(f"BIDS_ROOT={ _quote(bids_root) }")
        lines.append(f"FS_DIR={ _quote(fs_dir) }")
        lines.append(
            "T1W=$(find \"$BIDS_ROOT/$SUBJECT/anat\" -maxdepth 1 -type f \( -name '*_T1w.nii' -o -name '*_T1w.nii.gz' \) | head -n 1)"
        )
        lines.append("if [[ -z \"$T1W\" ]]; then echo 'T1w not found for $SUBJECT'; exit 1; fi")

        if mode == "fastsurfer":
            lines.append(
                f"singularity exec --nv {_quote(fastsurfer_image)} fastsurfer.sh --t1 \"$T1W\" --sid $SUBJECT --sd \"$FS_DIR\" --threads {slurm.get('cpus_per_task', 8)}"
            )
        else:
            freesurfer_image = config["fmriprep"].get("freesurfer_image")
            if freesurfer_image:
                lines.append(
                    f"singularity exec {_quote(freesurfer_image)} recon-all -s $SUBJECT -i \"$T1W\" -sd \"$FS_DIR\" -all"
                )
            else:
                lines.append(
                    "recon-all -s $SUBJECT -i \"$T1W\" -sd \"$FS_DIR\" -all"
                )

        pre_job.write_text("\n".join(lines) + "\n")
        scripts.append(str(pre_job))

    fmriprep_job = work_dir / "slurm_fmriprep.sh"
    fmriprep_output = work_dir / "slurm_fmriprep_%A_%a.out"
    header = _slurm_header("slurm_fmriprep", slurm, array_size, _quote(fmriprep_output))
    lines = [header]
    lines.append(f"SUBJECT=$(sed -n '$((SLURM_ARRAY_TASK_ID + 1))p' { _quote(subject_list_file) })")
    lines.append("if [[ -z \"$SUBJECT\" ]]; then echo 'Subject not found'; exit 1; fi")
    lines.append(f"BIDS_ROOT={ _quote(bids_root) }")
    lines.append(f"OUTPUT_DIR={ _quote(Path(config['fmriprep']['output_dir'])) }/$SUBJECT")
    lines.append(f"WORK_DIR={ _quote(Path(config['fmriprep']['work_dir'])) }/$SUBJECT")
    lines.append(f"FS_DIR={ _quote(fs_dir) }")
    lines.append(f"LICENSE_FILE={ _quote(license_file) }")

    command = [
        "singularity",
        "exec",
        "--cleanenv",
        _quote(fmriprep_image),
        "fmriprep",
        '"$BIDS_ROOT"',
        '"$OUTPUT_DIR"',
    ]
    if extra_args:
        command.append(extra_args)
    else:
        command.extend(["--participant-label", "$SUBJECT"])

    mode_args = []
    if mode == "none":
        mode_args.append("--fs-no-reconall")
    elif mode in ("reuse", "freesurfer", "fastsurfer"):
        mode_args.extend(["--fs-subjects-dir", '"$FS_DIR"', "--fs-no-reconall", "--fs-no-resume"])
    elif mode == "integrated" and license_file:
        mode_args.extend(["--fs-license-file", '"$LICENSE_FILE"'])

    if _find_multi_echo(subjects[0], config):
        mode_args.append("--me-output-echoes")

    command.extend(mode_args)
    lines.append(" ".join(command))
    fmriprep_job.write_text("\n".join(lines) + "\n")
    scripts.append(str(fmriprep_job))

    return scripts
