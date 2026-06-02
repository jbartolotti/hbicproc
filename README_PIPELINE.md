# HBICProc Pipeline

Minimal Python package for orchestrating an fMRI preprocessing workflow.

## Commands

Run one step for a subject:

```bash
python -m pipeline.cli run --subject sub-001 --step bids
python -m pipeline.cli run --subject sub-001 --step mriqc
python -m pipeline.cli run --subject sub-001 --step fmriprep
```

## Configuration

Edit `pipeline_config.json` to set study-specific paths and container images.

## Example placeholder tool commands

- `bidskit`

```bash
bidskit --input-dir sourcedata --output-dir work/bidskit --subject sub-001
```

- `MRIQC` via Singularity/Docker

```bash
singularity run docker://poldracklab/mriqc:latest work/bidskit derivatives/mriqc participant --participant_label sub-001
```

- `fMRIPrep` via Singularity/Docker

```bash
singularity run docker://poldracklab/fmriprep:latest work/bidskit derivatives/fmriprep --fs-license-file /path/to/license.txt participant --participant_label sub-001
```

## Design

- Each step is implemented in its own module under `pipeline/steps/`.
- Steps check whether expected outputs already exist and skip execution if present.
- Execution results are appended to a JSON log file in `logs/`.
- The package is intentionally lightweight and relies on subprocess calls instead of workflow frameworks.

## Protocol annotation tool

Run the annotation helper for a `Protocol_Translator.json` file:

```bash
python -m pipeline.annotate_protocol /path/to/Protocol_Translator.json
```

When installed, the package also exposes a convenience CLI entrypoint:

```bash
annotate_protocol /path/to/Protocol_Translator.json
```
