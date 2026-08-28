import shlex
from pathlib import Path

from ...core.paths import get_bids_root, ensure_dir
from .service import _resolve_freesurfer_subjects_dir, _resolve_fmriprep_image, _find_multi_echo


def _get_precompute_targets(subjects, config):
    bids_root = get_bids_root(config)
    configured_session = config["fmriprep"].get("precompute_session", "").strip()
    if configured_session and not configured_session.startswith("ses-"):
        configured_session = f"ses-{configured_session}"

    all_session_dirs = sorted({
        path.name
        for subject in subjects
        for path in (bids_root / subject).iterdir()
        if path.is_dir() and path.name.startswith("ses-")
    })

    selected_session = configured_session or (all_session_dirs[0] if all_session_dirs else None)

    targets = []
    for subject in subjects:
        targets.append((subject, selected_session))
    return targets


def _slurm_header(name, slurm, array_length, output_file, mode, email=None):
    is_precompute = name == "slurm_fmriprep_precompute"
    cpus_per_task = slurm.get("precompute_cpus_per_task", slurm.get("cpus_per_task", 8)) if is_precompute else slurm.get("cpus_per_task", 8)
    mem = slurm.get("precompute_mem", slurm.get("mem", "24G")) if is_precompute else slurm.get("mem", "24G")
    time_limit = (
        slurm.get("precompute_time", slurm.get("time_fastsurfer", "02:00:00"))
        if is_precompute
        else slurm.get("time_fmriprep", "06:00:00")
    )

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={slurm.get('partition', 'sixhour')}",
        f"#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --array=0-{max(array_length - 1, 0)}",
        f"#SBATCH --output={output_file}",
    ]
    if mode == "fastsurfer":
        lines.extend(["#SBATCH --gres=gpu:1",
                     f"#SBATCH --constraint={slurm.get('gpu_type', 'a100')}"])
    if email:
        lines.extend([
            "#SBATCH --mail-type=BEGIN,END,FAIL",
            f"#SBATCH --mail-user={email}",
        ])
    lines.extend([
        "set -euo pipefail",
        "pwd; hostname; date",
        "",
        "module load StdEnv singularity",
        "",
    ])
    return "\n".join(lines)


def _quote(path):
    return shlex.quote(str(path))


def generate_slurm_scripts(subjects, config):
    # Generate SLURM scripts for fMRIPrep and optional FreeSurfer precomputation.
    script_dir = Path(config.get("code_dir", "."))
    ensure_dir(script_dir)

    subject_list_file = script_dir / "slurm_fmriprep_subjects.txt"
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
        precompute_targets = _get_precompute_targets(subjects, config)
        precompute_list_file = script_dir / "slurm_fmriprep_precompute_targets.txt"
        precompute_list_file.write_text("\n".join(
            f"{subject}|{session}" if session else subject
            for subject, session in precompute_targets
        ) + "\n")

        pre_job = script_dir / "slurm_fmriprep_precompute.sh"
        pre_output = "slurm_fmriprep_precompute_%A_%a.out"
        header = _slurm_header(
            "slurm_fmriprep_precompute",
            slurm,
            len(precompute_targets),
            _quote(pre_output),
            mode,
            config.get("email", "")
        )
        lines = [header]
        lines.append(f"ENTRY=$(sed -n '$((SLURM_ARRAY_TASK_ID + 1))p' {_quote(precompute_list_file)})")
        lines.append("if [[ -z \"$ENTRY\" ]]; then echo 'Precompute entry not found'; exit 1; fi")
        lines.append("SUBJECT=${ENTRY%%|*}")
        lines.append("SESSION=${ENTRY#*|}")
        lines.append("if [[ \"$SESSION\" == \"$ENTRY\" ]]; then SESSION=''; fi")
        lines.append("if [[ -n \"$SESSION\" ]]; then T1W_FILE=\"/data/$SUBJECT/$SESSION/anat/${SUBJECT}_${SESSION}_T1w.nii.gz\"; else T1W_FILE=\"/data/$SUBJECT/anat/${SUBJECT}_T1w.nii.gz\"; fi")
        lines.append("if [[ ! -f \"$T1W_FILE\" ]]; then echo \"T1w file not found: $T1W_FILE\"; exit 1; fi")
        lines.append("")
        lines.append(f"BIDS_ROOT={_quote(bids_root)}")
        lines.append(f"FS_DIR={_quote(fs_dir)}")
        lines.append("")

        if mode == "fastsurfer":
            lines.append(f"""singularity exec --cleanenv --nv \\
                -B $BIDS_ROOT:/data \\
                -B $FS_DIR:/fs \\
                -B {_quote(license_file)}:$HOME/license.txt \\
                -B $SCRATCH:/tmp \\
                {_quote(fastsurfer_image)}  \\
                /fastsurfer/run_fastsurfer.sh \\
                --fs_license $HOME/license.txt \\
                --t1 \"$T1W_FILE\" \\
                --sid $SUBJECT \\
                --sd /fs \\
                --threads {str(slurm.get('cpus_per_task', 8))}
            """)
        else:
            freesurfer_image = config["fmriprep"].get("freesurfer_image")
            if freesurfer_image:
                lines.append(f"""export SUBJECTS_DIR=/fs
                    export SINGULARITYENV_SUBJECTS_DIR=/fs
                    export APPTAINERENV_SUBJECTS_DIR=/fs

                    singularity exec --cleanenv \\
                    -B $BIDS_ROOT:/data \\
                    -B {_quote(license_file)}:/opt/freesurfer/license.txt \\
                    -B $SCRATCH:/tmp \\
                    {_quote(freesurfer_image)} \" \\
                    recon-all \\
                    -s $SUBJECT \\
                    -fs_license $HOME/license.txt \\
                    -i \"$T1W_FILE\" \\
                    -sd /fs \\
                    -all
                """)

        pre_job.write_text("\n".join(lines) + "\n")
        scripts.append(str(pre_job))

    fmriprep_job = script_dir / "slurm_fmriprep.sh"
    fmriprep_output = "slurm_fmriprep_%A_%a.out"
    header = _slurm_header(
        "slurm_fmriprep",
        slurm,
        array_size,
        _quote(fmriprep_output),
        mode,
        config.get("email", ""),
    )
    lines = [header]
    lines.append(f"SUBJECT=$(sed -n '$((SLURM_ARRAY_TASK_ID + 1))p' {_quote(subject_list_file)})")
    lines.append("if [[ -z \"$SUBJECT\" ]]; then echo 'Subject not found'; exit 1; fi")
    lines.append("")
    lines.append("export APPTAINERENV_FS_LICENSE=/opt/fslicense/license.txt")
    lines.append("export SINGULARITYENV_FS_LICENSE=/opt/fslicense/license.txt")
    lines.append(f"BIDS_ROOT={_quote(bids_root)}")


    # Build fMRIPrep command
    cmd = f"""singularity exec --cleanenv \\
    -B $BIDS_ROOT:/data \\
    -B {_quote(license_file)}:/opt/fslicense \\
    {_quote(fmriprep_image)} \\
    /data /data/derivatives participant \\
    --participant-label "$SUBJECT" \\
    --nthreads {slurm.get('cpus_per_task', 8)} \\
    --omp-nthreads {slurm.get('cpus_per_task', 8)} \\"""

    # Mode-specific arguments
    mode_args = []

    if mode == "none":
        mode_args.append("--fs-no-reconall")

    elif mode in ("reuse", "freesurfer", "fastsurfer"):
        mode_args.extend([
            "--fs-subjects-dir /data/derivatives/freesurfer",
            "--fs-no-reconall"
        ])

    # Multi-echo
    if _find_multi_echo(subjects[0], config):
        mode_args.append("--me-output-echos")

    # Append mode args
    for arg in mode_args:
        cmd += f"\n    {arg} \\"

    # Remove trailing backslash safely
    cmd = cmd.rstrip(" \\") + "\n"

    lines.append(cmd)


    fmriprep_job.write_text("\n".join(lines) + "\n")
    scripts.append(str(fmriprep_job))

    return scripts
