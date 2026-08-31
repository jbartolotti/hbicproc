# Agent Guide: hbicproc Pipeline Architecture

This file documents the architectural rules for the `hbicproc` package so new features (especially new
pipeline stages) can be added consistently. Read this before adding or modifying stages, CLI commands,
or processing logic.

## Layering (strict, one-way dependency direction)

```
CLI (pipeline/cli.py)
  -> Runner (pipeline/runner.py: PipelineRunner)
    -> Stages (pipeline/stages/*.py — thin wrappers)
      -> Processing services (pipeline/processing/<domain>/service.py)
        -> Core helpers (pipeline/core/paths.py, subprocess_utils.py, container.py)
```

- **CLI** parses arguments and calls `PipelineRunner`. It must not contain business logic.
- **Runner** (`PipelineRunner`) sequences stages, loads/saves subject state, and resolves "next eligible
  subject." It must not import processing services directly — only stage classes.
- **Stages** (`pipeline/stages/*.py`) are thin adapters: they call a processing service's `run(...)`
  function and translate its return value into a `StageResult`. Stages hold no real logic themselves.
- **Processing services** (`pipeline/processing/<domain>/service.py`) contain the actual work (building
  commands, calling subprocesses, parsing files, etc.). Each domain may have extra modules alongside
  `service.py` (e.g. `slurm.py`, `xnat_client.py`, `exclusions.py`) for large sub-concerns.


## Adding a new stage — required steps

1. Create `pipeline/processing/<name>/service.py` with a public `run(subject, config, dry_run=False,
   rerun=False)` function. Return either a `dict` (with at least `success`, and optionally `skipped`,
   `message`, `next_command`) or construct a `StageResult` directly if you prefer.
2. Create `pipeline/stages/<name>.py` defining `class <Name>Stage(BaseStage)`:
   - Set `name = "<name>"` (must match the processing package name and the CLI subcommand).
   - Set `state_key = "<name>_complete_or_similar"` — the subject-state boolean flag that marks this
     stage done. Omit/`None` only if the stage should never be considered "complete" (rare).
   - Set `human_step = True` only if a human must act out-of-band before the stage can be marked
     complete (see `qc_review.py` — `BaseStage.execute()` will not auto-set `state_key` for
     `human_step` stages even on success).
   - Implement `run(self, subject, config, state, dry_run=False, rerun=False)` that calls the
     processing service and converts its result into a `StageResult`.
3. Register the stage in `pipeline/stages/__init__.py`'s `STAGE_CLASSES` dict:
   `"<name>": <Name>Stage`. **This single registration is what makes the CLI generate a working
   `hbicproc <name>` subcommand automatically** (see "CLI wiring" below) — no changes to `cli.py` are
   needed for a standard stage.
4. Decide whether the stage belongs in `PipelineRunner.stage_order` (in `pipeline/runner.py`):
   - Add it there if the stage is part of the strictly-ordered, resumable pipeline (i.e. `hbicproc run`
     / `hbicproc <stage>` with no subject should be able to auto-select the next eligible subject for
     it, and completion should gate later stages).
   - Leave it out if the stage is optional/independent (like `behavior`) and shouldn't block or be
     blocked by the ordered pipeline. Optional stages must always be invoked with an explicit subject
     or `--all`; the CLI enforces this generically via
     `if stage_name not in PipelineRunner.stage_order: parser.error(...)`.
   - `STAGE_CLASSES` (the full registry) and `stage_order` (the ordered resumable subset) are
     deliberately different collections — do not assume every registered stage is in `stage_order`.
5. If the stage needs extra CLI-only flags (like `download --summary` or `behavior --task`), add them
   in `_build_parser()` in `cli.py` guarded by `if stage_name == "<name>":`, and handle the flag inside
   `_handle_stage()`. Keep this minimal — most stages need no extra flags.

## CLI wiring (`pipeline/cli.py`)

- `cli.py` builds one argparse subcommand per entry in `STAGE_CLASSES` (imported from `.stages`) inside
  `_build_parser()`. It does **not** hardcode a list of stage names — this is the "single dispatcher
  driven by `STAGE_CLASSES`" principle. Do not reintroduce a hardcoded stage-name list.
- Every generated stage subcommand supports `subject` (optional), `--all`, and `--rerun` uniformly.
- Non-stage commands (`init`, `status`, `run`, `exclude`) are separate, explicitly-defined subcommands
  handled by their own `_handle_*` function in `main()`'s dispatch.
- Batch/utility functions that don't fit the per-subject stage model (e.g. `download_all`,
  `summarize_downloads` in `pipeline/processing/download/service.py`) are imported directly inside
  `_handle_stage()` and are legitimate — they are not considered a disallowed "bypass" since no other
  layer exposes that functionality. New stages should avoid needing this unless truly necessary.

## Stage contract details

- `BaseStage.execute()` (in `pipeline/stages/base.py`) is the entry point the runner calls — never call
  a stage's `run()` directly from the CLI or runner. `execute()`:
  1. Returns an early `skipped=True` `StageResult` if `state_key` is already `True` in subject state and
     `rerun` is not set.
  2. Otherwise calls `run(...)`.
  3. On success (and not skipped, and not `human_step`), sets `state[state_key] = True`. The caller
     (`PipelineRunner.run_stage`/`run_resume`) is responsible for persisting that state via
     `save_subject_state`.
- `StageResult` fields: `success: bool`, `message: str`, `skipped: bool = False`,
  `needs_review: bool = False`, `next_command: Optional[str] = None`, `details: dict = {}`. Put any
  extra structured data (subprocess command string, output paths, etc.) in `details`, not as new
  top-level fields.

## Configuration (`pipeline/config.py`)

- `load_config()` reads `pipeline_config.json`, applies defaults per top-level section (`xnat`,
  `bidskit`, `mriqc`, `fmriprep`, `hbicproc`, `behavior`, ...), then resolves relative paths against the
  config file's directory. When adding a new stage with its own config section, add a matching default
  block and include the section name in both the defaults-merge loop and the path-resolution loop in
  `config.py`.

## Core helpers (`pipeline/core/`)

- `paths.py`: subject/session directory resolution, `list_subjects`, JSON load/write helpers. Prefer
  these over ad hoc `Path` construction so subject discovery stays consistent across stages.
- `subprocess_utils.py`: wrappers for running external commands (used for dry-run-aware command
  execution — check here before writing a new subprocess-invocation helper).
- `container.py`: Singularity/container image resolution helpers.

## Testing and verification

- Run `pytest` from the repo root (venv at `.venv/`) for unit tests (currently covers the behavior
  subsystem in `tests/test_behavior.py`).
- `python -c "import pipeline.cli"` is a fast sanity check that catches import-time errors after
  restructuring.
- `hbicproc <stage> --help` / `hbicproc --help` diffs are the easiest way to confirm CLI wiring changes
  behave as expected after modifying `STAGE_CLASSES` or `cli.py`.
- Prefer `--dry-run` smoke tests against a real BIDS dataset over inventing new test fixtures when
  validating end-to-end stage wiring.

## Anti-patterns to avoid

- Do not add per-stage `if`/`elif` branches in `cli.py` for standard stage dispatch — that defeats the
  `STAGE_CLASSES`-driven design this package was refactored toward.
- Do not put business logic directly in a `pipeline/stages/*.py` file — it belongs in
  `pipeline/processing/<domain>/service.py`.
- Do not recreate a `pipeline/steps/` bypass package.
- Do not assume `STAGE_CLASSES` and `PipelineRunner.stage_order` are the same set.
