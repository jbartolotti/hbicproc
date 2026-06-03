import argparse
import sys
from pathlib import Path

from .config import load_config, save_default_config
from .pipeline import PipelineRunner
from .state import load_subject_state, save_subject_state
from .utils import list_subjects, write_json, load_json


def _print_result(result):
    if result.skipped:
        print(f"SKIPPED: {result.message}")
        return

    print(result.message)
    if result.next_command:
        print("\nNext step:")
        print(f"  {result.next_command}")
    if result.details:
        if result.details.get("command"):
            print(f"\nCommand:\n  {result.details['command']}")


def _load_config(path):
    try:
        return load_config(path)
    except Exception as exc:
        print(f"ERROR: Failed to load config: {exc}", file=sys.stderr)
        raise


def _write_default_config(path):
    save_default_config(path)
    print(f"Created default pipeline config at {path}")
    print("Edit the file to adjust study paths, containers, and credentials.")


def _run_for_all(stage_name, config, dry_run=False):
    subjects = list_subjects(config)
    if not subjects:
        print("No subjects found in BIDS output directory.")
        return 1
    runner = PipelineRunner(config)
    exit_code = 0
    for subject in subjects:
        print(f"\n=== {stage_name} {subject} ===")
        result = runner.run_stage(stage_name, subject, dry_run=dry_run)
        _print_result(result)
        if not result.success:
            exit_code = 1
    return exit_code


def _run_subject_stage(stage_name, subject, config, dry_run=False):
    runner = PipelineRunner(config)
    result = runner.run_stage(stage_name, subject, dry_run=dry_run)
    _print_result(result)
    return 0 if result.success else 1


def _run_resume(subject, config, dry_run=False):
    runner = PipelineRunner(config)
    result = runner.run_resume(subject, dry_run=dry_run)
    _print_result(result)
    return 0 if result.success else 1


def _subject_exclusion(subject, config, runs, clear):
    exclusions_file = Path(config["hbicproc"]["exclusions_file"])
    exclusions = load_json(exclusions_file, default={})
    if clear:
        if subject in exclusions:
            exclusions.pop(subject)
        save_exclusions = True
    else:
        subject_data = exclusions.setdefault(subject, {})
        excluded_runs = subject_data.setdefault("excluded_runs", [])
        for run in runs:
            if run not in excluded_runs:
                excluded_runs.append(run)
        subject_data["updated_at"] = str(Path().resolve())
        exclusions[subject] = subject_data
        save_exclusions = True

    if save_exclusions:
        exclusions_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(exclusions_file, exclusions)
        state = load_subject_state(Path(config["study_root"]) / subject)
        state["qc_reviewed"] = True
        save_subject_state(Path(config["study_root"]) / subject, state)

    print(f"Recorded exclusions for {subject} in {exclusions_file}")
    print("QC review is now complete for this subject.")
    print("Next step: hbicproc preprocess {subject}".format(subject=subject))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="hbicproc stage-based pipeline CLI")
    parser.add_argument(
        "--config",
        default="pipeline_config.json",
        help="Path to the pipeline config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing them.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a default pipeline config file.")
    init_parser.add_argument("path", help="Path to write a new config file.")

    for stage_name in ["download", "bidsify", "validate", "qc", "preprocess"]:
        stage_parser = subparsers.add_parser(stage_name, help=f"Run the {stage_name} stage for a subject.")
        stage_parser.add_argument("subject", nargs="?", help="Participant label, e.g. sub-011.")
        stage_parser.add_argument(
            "--all",
            action="store_true",
            help="Run the stage for all subjects found in the BIDS output directory.",
        )

    run_parser = subparsers.add_parser("run", help="Resume the pipeline from the last incomplete stage.")
    run_parser.add_argument("subject", help="Participant label, e.g. sub-011.")

    exclude_parser = subparsers.add_parser("exclude", help="Record MRIQC exclusions for a subject.")
    exclude_parser.add_argument("subject", help="Participant label, e.g. sub-011.")
    exclude_parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run label to exclude, e.g. task-nback_run-2. Can be repeated.",
    )
    exclude_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear recorded exclusions for the subject.",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        _write_default_config(args.path)
        return 0

    config = _load_config(args.config)

    if args.command in ["download", "bidsify", "validate", "qc", "preprocess"]:
        if args.all:
            if args.subject:
                parser.error("Cannot specify a subject and --all together.")
            return _run_for_all(args.command, config, dry_run=args.dry_run)
        if not args.subject:
            parser.error("Subject is required unless --all is used.")
        return _run_subject_stage(args.command, args.subject, config, dry_run=args.dry_run)

    if args.command == "run":
        return _run_resume(args.subject, config, dry_run=args.dry_run)

    if args.command == "exclude":
        if not args.runs and not args.clear:
            parser.error("You must provide --run or --clear.")
        return _subject_exclusion(args.subject, config, args.runs or [], args.clear)

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
