# HBICProc Pipeline

A stage-based pipeline orchestrator for neuroimaging workflows with human checkpoints and resumable state.

## Commands

Stage-based CLI usage:

```bash
hbicproc init pipeline_config.json
hbicproc download sub-001
hbicproc bidsify sub-001
hbicproc validate sub-001
hbicproc qc sub-001
hbicproc exclude sub-001 --run task-nback_run-2
hbicproc preprocess sub-001
hbicproc behavior sub-001 --task stroop
hbicproc run sub-001
hbicproc status
```

Batch commands:

```bash
hbicproc qc --all
hbicproc preprocess --all
hbicproc behavior --all
hbicproc run --all
```

Stage commands also support `--rerun` to force execution even when a stage is marked complete.

## Pipeline stages

1. `download`
2. `bidsify`
3. `validate`
4. `qc`
5. `qc_review` (human step via `exclude`)
6. `preprocess` (fMRIPrep, with optional FreeSurfer / FastSurfer integration)

The `behavior` stage is optional and independent of the ordered pipeline above; it can be run at any
time once E-Prime source files are available.

Each stage is a thin `pipeline.stages` wrapper around a corresponding `pipeline.processing.<stage>.service`
module. The CLI (`pipeline/cli.py`) builds its subcommands directly from the `STAGE_CLASSES` registry in
`pipeline/stages/__init__.py`, so every registered stage automatically gets a CLI subcommand.

## Configuration

Edit `pipeline_config.json` to set your study root, BIDS paths, Singularity images, and exclusions file.

A sample configuration is available at `pipeline_config.sample.json`.

### fMRIPrep configuration

The `fmriprep` section controls the `preprocess` stage, including optional FreeSurfer / FastSurfer integration.
fMRIPrep is not a separate CLI command; it runs as part of `hbicproc preprocess`.

- `fmriprep.singularity_image`: Singularity image reference for fMRIPrep.
- `fmriprep.fmriprep_image`: explicit Singularity image for fMRIPrep (falls back to `singularity_image`).
- `fmriprep.fastsurfer_image`: Singularity image reference for FastSurfer when using `freesurfer_mode: fastsurfer`.
- `fmriprep.freesurfer_image`: Singularity image reference for standalone FreeSurfer recon-all when using `freesurfer_mode: freesurfer`.
- `fmriprep.output_dir`: derivative output directory for fMRIPrep results.
- `fmriprep.work_dir`: temporary working directory for fMRIPrep and SLURM scripts.
- `fmriprep.freesurfer_subjects_dir`: directory for precomputed FreeSurfer / FastSurfer outputs.
- `fmriprep.fs_license_file`: path to the FreeSurfer license file.
- `fmriprep.extra_args`: extra fMRIPrep command-line arguments, formatted with `{subject}`.
- `fmriprep.use_slurm`: set to `true` to generate SLURM submission scripts instead of running locally.
- `fmriprep.freesurfer_mode`: one of `integrated`, `none`, `reuse`, `freesurfer`, or `fastsurfer`.
- `fmriprep.slurm`: SBATCH defaults used when `use_slurm` is enabled.

When `freesurfer_mode` is `fastsurfer`, FastSurfer is run first and then fMRIPrep uses the precomputed `freesurfer_subjects_dir`.
When `freesurfer_mode` is `freesurfer`, `recon-all` is run first and then fMRIPrep reuses the generated subjects directory.

### XNAT download configuration

The download stage now uses `pyxnat` and downloads data directly from XNAT.
Set XNAT details in the `xnat` section of `pipeline_config.json`:

- `xnat.server`: XNAT server URL.
- `xnat.project_id`: XNAT project identifier.
- `xnat.credentials_file`: path to a file containing XNAT credentials.
- `xnat.session_names_file`: optional TSV mapping XNAT scan labels to BIDS session labels.
- `xnat.session_names_delimiter`: optional delimiter used when a session label contains both subject and session identifiers.
- `xnat.verbose`: optional boolean to enable detailed XNAT download debug logging to the console.
- `xnat.output_dir`: local sourcedata directory.
- `xnat.verify_ssl`: whether to verify TLS certificates.

If `xnat.username` / `xnat.password` are not set and `xnat.credentials_file` is omitted, `hbicproc` will also attempt to read credentials from `~/.netrc` for the XNAT host.

The credentials file may be JSON or simple key/value text. For example:

```json
{"username": "MY_USER", "password": "MY_PASSWORD"}
```

or:

```
username=MY_USER
password=MY_PASSWORD
```

Alternatively, store credentials in a secure `~/.netrc` file (set file permisisons to only owner-read with 'chmod 600 ~/.netrc'):

```
machine xnat.kumc.edu login MY_USER password MY_PASSWORD
```

The session mapping file is used to manage inconsistent session naming. The file should include a header row. Prefer using separate `subject` and `session` columns whenever possible:

```
session_sic	subject	session
123456_001_T1	001	T1
123456_001_Time2	001	T2
```

If you instead have a combined session label such as `011_BL`, set `xnat.session_names_delimiter` to `_` in `pipeline_config.json`.

An example TSV file is included at `pipeline/session_names.example.tsv`.

If `xnat.session_names_file` is provided, `hbicproc` uses it to map XNAT experiments to BIDS session IDs. If it is omitted, the pipeline will attempt to derive `ses-...` labels automatically from the XNAT experiment labels.

Run `hbicproc download sub-001` to download all sessions for `sub-001` into:

```
sourcedata/sub-001/ses-T1
sourcedata/sub-001/ses-T2
```

## MRIQC

The QC stage runs participant-level MRIQC and prints the exact next command for human review.

## Behavioral events configuration

The `behavior` stage uses the `behavior` section of `pipeline_config.json` to locate E-Prime files, select task parsers, and write BIDS-style events outputs.

- `behavior.task_configs_dir`: directory containing JSON task configuration files for each supported task. The default example is `code/behavior`.
- `behavior.parser_plugins_dir`: optional directory containing additional parser plugin modules for custom tasks. Files are loaded automatically when present.
- `behavior.edat3_search_root`: root directory used to search for source `.edat3` files. The default is `sourcedata`.
- `behavior.output_dir`: directory where generated `task-*_events.tsv` and `task-*_events.json` files will be written.
- `behavior.reader_backend`: backend selector for the E-Prime reader. Use `auto` for automatic detection, or set it to a specific module name such as `edat` or `pandas`.

Example:

```json
"behavior": {
  "task_configs_dir": "code/behavior",
  "parser_plugins_dir": "code/behavior/plugins",
  "edat3_search_root": "sourcedata",
  "output_dir": "derivatives/behavior/events",
  "reader_backend": "auto"
}
```

The `behavior` stage expects task configuration JSON files in the configured task config directory. A worked example is included at `code/behavior/stroop.json`.

## Status reporting

`hbicproc status` scans all recorded subject state files and writes a stage-completion grid to an SVG file in the BIDS code directory by default.

## Exclusions

User decisions are stored in `derivatives/hbicproc/exclusions.json` and the `preprocess` stage reads them before execution.

## Protocol annotation tool

Run the annotation helper for a `Protocol_Translator.json` file:

```bash
python -m pipeline.annotate_protocol /path/to/Protocol_Translator.json
```

When installed, the package also exposes a convenience CLI entrypoint:

```bash
annotate_protocol /path/to/Protocol_Translator.json
```
