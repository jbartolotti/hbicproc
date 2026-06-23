from pathlib import Path
import shutil

from .base import BaseStage, StageResult
from ..container import SingularityRunner
from ..utils import ensure_dir, get_bids_root


class QcStage(BaseStage):
    name = "qc"
    state_key = "qc_complete"

    def run(self, subject, config, state, dry_run=False, rerun=False):
        bids_root = get_bids_root(config)
        bids_dir = bids_root / subject
        if not bids_dir.exists():
            return StageResult(success=False, message=f"BIDS subject directory not found: {bids_dir}")

        image = config["mriqc"]["singularity_image"]
        if not any(image.startswith(prefix) for prefix in ("docker://", "shub://", "library://")):
            if not Path(image).exists():
                return StageResult(success=False, message=f"Singularity image not found: {image}")

        output_dir = Path(config["mriqc"]["output_dir"])
        subject_output_dir = output_dir / subject
        base_work_dir = Path(config["mriqc"]["work_dir"])
        work_dir = base_work_dir / f"work_{subject}"

        if rerun and subject_output_dir.exists() and not dry_run:
            shutil.rmtree(subject_output_dir)

        ensure_dir(output_dir)
        ensure_dir(work_dir)

        internal_input_dir = Path("/bids_root")
        internal_work_dir = Path("/work")

        binds = [
            (str(bids_root.resolve()), str(internal_input_dir)),
            (str(work_dir.resolve()), str(internal_work_dir)),
        ]

        try:
            internal_output_dir = Path("/bids_root") / output_dir.relative_to(bids_root)
        except ValueError:
            internal_output_dir = Path("/output")
            binds.append((str(output_dir.resolve()), str(internal_output_dir)))

        extra_args = config["mriqc"].get("extra_args", "participant --participant_label {subject}").format(subject=subject)
        if "-w" not in extra_args and "--work-dir" not in extra_args:
            extra_args = f"{extra_args} -w {internal_work_dir}"

        binds = [
            (str(bids_root.resolve()), str(internal_input_dir)),
            (str(work_dir.resolve()), str(internal_work_dir)),
        ]

        print(f"QC stage for {subject}")
        print(f"  BIDS root: {bids_root}")
        print(f"  MRIQC output directory: {output_dir}")
        print(f"  MRIQC work directory: {work_dir}")
        print(f"  Binding BIDS root to container path {internal_input_dir}")
        print(f"  Binding work dir to container path {internal_work_dir}")

        runner = SingularityRunner(image, clean_env=True)
        result = runner.run(
            extra_args=(f"{internal_input_dir} {internal_output_dir} {extra_args}"),
            dry_run=dry_run,
            binds=binds,
            clean_env=True,
        )

        if not result["success"]:
            return StageResult(success=False, message=result.get("message", "MRIQC failed."), details=result)

        report_files = sorted(subject_output_dir.rglob("*.html"))
        if not report_files:
            return StageResult(
                success=False,
                message=(
                    f"MRIQC completed but no HTML reports were found under {output_dir}.\n"
                    "Check the MRIQC output directory and container logs."
                ),
                details={**result, "output_dir": str(output_dir)},
            )

        report_list = [str(path) for path in report_files[:5]]
        message = (
            f"MRIQC participant stage completed for {subject}.\n"
            f"Review HTML reports in: {output_dir}\n"
            f"Most likely report: {report_list[0]}\n\n"
            "Then run: hbicproc qc_review {subject}"
        )

        return StageResult(
            success=True,
            message=message,
            next_command=f"hbicproc qc_review {subject}",
            details={
                "command": result.get("command"),
                "output_dir": str(output_dir),
                "reports": report_list,
            },
        )
