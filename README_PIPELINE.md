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
hbicproc run sub-001
hbicproc status
```

Batch commands:

```bash
hbicproc qc --all
hbicproc preprocess --all
hbicproc run --all
```

Stage commands also support `--rerun` to force execution even when a stage is marked complete.

## Pipeline stages

1. `download`
2. `bidsify`
3. `validate`
4. `qc`
5. `qc_review` (human step via `exclude`)
6. `preprocess`

## Configuration

Edit `pipeline_config.json` to set your study root, BIDS paths, Singularity images, and exclusions file.

A sample configuration is available at `pipeline_config.sample.json`.

## MRIQC

The QC stage runs participant-level MRIQC and prints the exact next command for human review.

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
