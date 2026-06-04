from pathlib import Path

from .base import BaseStage, StageResult
from ..container import SingularityRunner
from ..utils import ensure_dir, load_json, get_bids_root


class PreprocessStage(BaseStage):
    name = "preprocess"
    state_key = "preprocessed"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        bids_dir = get_bids_root(config)
        if not bids_dir.exists():
            return StageResult(success=False, message=f"BIDS directory not found: {bids_dir}")

        output_dir = Path(config["fmriprep"]["output_dir"]) / subject
        work_dir = Path(config["fmriprep"]["work_dir"]) / subject
        license_file = Path(config["fmriprep"]["fs_license_file"])
        ensure_dir(output_dir)
        ensure_dir(work_dir)

        if not license_file.exists():
            return StageResult(
                success=False,
                message=(
                    f"FreeSurfer license file not found: {license_file}."
                    " Set `fmriprep.fs_license_file` in your config before preprocessing."
                ),
            )

        exclusions = load_json(config["hbicproc"]["exclusions_file"], default={})
        subject_exclusions = exclusions.get(subject, {}).get("excluded_runs", [])
        exclusion_note = ""
        if subject_exclusions:
            exclusion_note = f"Detected excluded runs: {', '.join(subject_exclusions)}. "

        extra_args = config["fmriprep"].get("extra_args", "participant --participant_label {subject}").format(subject=subject)
        if "--fs-license-file" not in extra_args:
            extra_args = f"--fs-license-file {license_file} {extra_args}"

        runner = SingularityRunner(config["fmriprep"]["singularity_image"])
        result = runner.run(
            input_dir=bids_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            extra_args=extra_args,
            dry_run=dry_run,
        )

        if not result["success"]:
            return StageResult(success=False, message=result.get("message", "fMRIPrep failed."), details=result)

        return StageResult(
            success=True,
            message=(
                f"fMRIPrep completed for {subject}.\n{exclusion_note}"
                f"Output written to {output_dir}."
            ),
            details={"command": result.get("command"), "output_dir": str(output_dir)},
        )
