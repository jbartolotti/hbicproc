import argparse
import sys
from .config import load_config
from .logger import append_event, append_log
from .steps import download, bids, mriqc, fmriprep

STEPS = {
    "download": download.run,
    "bids": bids.run,
    "mriqc": mriqc.run,
    "fmriprep": fmriprep.run,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a minimal fMRI preprocessing pipeline step for a single subject."
    )
    parser.add_argument("command", choices=["run"], help="Pipeline command to execute.")
    parser.add_argument("--subject", required=True, help="Participant label, e.g. sub-001.")
    parser.add_argument(
        "--step",
        required=True,
        choices=STEPS,
        help="Pipeline step to execute: download, bids, mriqc, fmriprep.",
    )
    parser.add_argument(
        "--config",
        default="pipeline_config.json",
        help="Path to a JSON pipeline config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the command that would run without executing it.",
    )

    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error("Only the 'run' command is supported.")

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"ERROR: Failed to load config: {exc}", file=sys.stderr)
        sys.exit(2)

    append_event(f"Loaded config from {args.config}", config, subject=args.subject, step="config")
    runner = STEPS[args.step]
    print(f"Running step '{args.step}' for subject '{args.subject}'")
    append_event(f"Starting step {args.step}", config, subject=args.subject, step=args.step)
    status = runner(args.subject, config, dry_run=args.dry_run)
    log_path = append_log(args.subject, args.step, status, config)
    append_event(f"Completed step {args.step} with status {status.get('message')}", config, subject=args.subject, step=args.step)

    if args.dry_run:
        print(f"Dry run completed, log entry saved to {log_path}")
        return 0

    if status["success"]:
        if status.get("skipped"):
            print(f"Step {args.step} skipped: {status['message']}")
        else:
            print(f"Step {args.step} completed successfully.")
        return 0

    print(f"Step {args.step} failed: {status['message']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
