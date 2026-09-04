# Subject-level analysis

The `analysis` stage uses task-centered JSON configuration. Each enabled task declares
one or more reusable models and one or more analyses that reference those models.

## Configuration

The global `analysis.input_dataset` is inherited by every task. A task may override it
with its own dataset name and path. Paths are resolved by the normal configuration
loader.

```json
{
  "analysis": {
    "output_dir": "derivatives/hbicproc",
    "input_dataset": {
      "name": "fmriprep",
      "path": "derivatives/fmriprep"
    },
    "subject": {
      "tasks": {
        "nback": {
          "enabled": true,
          "models": {
            "canonical_glm": {
              "type": "canonical_glm",
              "t_r": 2.0,
              "hrf_model": "spm",
              "high_pass": 0.01,
              "confounds": ["trans_x", "trans_y"]
            }
          },
          "analyses": {
            "activation": {
              "model": "canonical_glm",
              "contrasts": {
                "2back_gt_1back": "twoback - oneback"
              },
              "output_types": ["effect_size", "z_score"]
            }
          }
        }
      }
    }
  }
}
```

Tasks and analyses default to enabled. Set `enabled` to `false` to disable one.
Required confounds must be present in the selected derivative confounds file.
Missing required columns produce a run-specific validation error.

## Inputs

Raw BIDS data supplies events. The configured preprocessed input dataset supplies
BOLD images and confounds. Models are fitted independently for each discovered run.

A `TaskRunContext` identifies the subject, task, optional session and run, input files,
and selected input dataset. The context is deliberately independent of model and
analysis state so combined-run support can be added later.

## Outputs

Model and analysis products use separate run-specific namespaces:

```text
derivatives/hbicproc/
  sub-001/
    ses-01/
      func/
        task-nback/
          run-1/
            models/
              canonical_glm/
                sub-001_ses-01_task-nback_run-1_desc-model-metadata.json
                sub-001_ses-01_task-nback_run-1_desc-design-matrix.tsv
            analyses/
              activation/
                sub-001_ses-01_task-nback_run-1_desc-first-level-report.html
                sub-001_ses-01_task-nback_run-1_desc-2back-gt-1back_stat-effect.nii.gz
                sub-001_ses-01_task-nback_run-1_desc-2back-gt-1back_stat-z.nii.gz
```

The fitted GLM computes requested activation contrasts immediately while the fitted
model remains in memory. The model derivative includes portable metadata and the
design matrix; Nilearn fitting state is not serialized. Additional contrasts require
refitting the model. Contrast configuration is not part of the model fingerprint.

Atlas settings are analysis concerns and are excluded from model fingerprints. Atlas
source files are fetched into the external `~/.cache/hbicproc/atlases` cache by default,
not into the BIDS tree. Resampled atlas products belong to the analysis namespace.

## Commands

The existing commands remain available:

```text
hbicproc analysis sub-001
hbicproc analysis --all
```

The `analysis` service validates configuration, discovers task runs, fits supported
models, writes model derivatives, and executes configured activation outputs. Group
analysis is not implemented.
