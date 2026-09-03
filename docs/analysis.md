# Subject-level analysis stage

The analysis feature is implemented as a normal pipeline stage rather than as a standalone CLI bypass. The stage wrapper sits in the package’s usual layering:


## Stage contract


## Plugin registry

The plugin layer remains available behind the processing service for extensibility:


This keeps the scientific logic modular without breaking the project’s stage-first architecture.

## Dataset index / BIDS discovery layer

Analysis discovery is intentionally centralized in a shared dataset index instead of being reimplemented in each plugin. The core abstraction lives in `pipeline/processing/analysis/dataset.py` and exposes an `AnalysisRun` dataclass plus a `DatasetIndex` helper built on PyBIDS.

A typical call path looks like this:

```python
from pipeline.processing.analysis.dataset import DatasetIndex

dataset = DatasetIndex.from_config(config)
runs = dataset.get_task_runs(subject="001", task="rest")

for run in runs:
    print(run.subject, run.session, run.run, run.bold_path)
```

This keeps plugin code focused on scientific processing and avoids repeated recursive filesystem scans or BIDS entity parsing in each analysis method.

The dataset index uses PyBIDS to build a BIDSLayout for the configured dataset root and resolves run-level metadata with subject/task/session/run matching from the indexed layout rather than by custom `rglob` traversal.

## Activation plugin requirements

The activation plugin is intentionally generic and configuration-driven. It does not carry task-specific or study-specific defaults.

### Explicit contrasts

The plugin requires a configuration entry for `analysis.subject.activation.contrasts` and fails validation if it is empty or missing. The design matrix is not inferred from a trial type column and no default reference contrast is generated.

### Config-driven confounds

Confound specification lives in configuration and is expanded at runtime. For example:

```json
{
  "analysis": {
    "subject": {
      "activation": {
        "enabled": true,
        "tasks": ["rest"],
        "contrasts": {
          "condition_a_minus_baseline": "condition_a - baseline"
        },
        "confounds": {
          "motion": true,
          "motion_derivatives": true,
          "framewise_displacement": true,
          "acompcor": 5
        }
      }
    }
  }
}
```

This expands to motion and derivative regressors plus the configured number of aCompCor components without hard-coded study choices in the plugin.

### Session handling

Session values are stored internally without BIDS prefixes, for example `baseline` rather than `ses-baseline`. Prefixes such as `ses-` and `run-` are added only by the derivative path helpers when building final BIDS-compliant output names.

### Nilearn-native behavior

The plugin relies on Nilearn APIs whenever they provide the required functionality:


### Output naming

Output files are written as BIDS-style derivative names with entity labels such as `sub-*`, `ses-*`, `task-*`, `run-*`, `desc-*`, and `stat-*`. Condition outputs use effect-map naming and contrast variance outputs use Nilearn effect-variance images.

## Future analyses

New analysis methods should follow the same pattern: add a plugin class under `pipeline/processing/analysis/...`, keep it isolated from the stage wrapper, and expose it through the `analysis` stage so it integrates with the existing pipeline conventions.

# Subject-level analysis architecture

The analysis stage is a user-facing pipeline stage exposed as `hbicproc analysis`.
It uses a task-centered configuration and separates reusable model derivatives from
scientific analysis products:

```text
task -> model specification -> model derivatives -> analysis specification -> analysis derivatives
```

The model layer will own fitting and reusable derivatives. The analysis layer will
consume those derivatives and write analysis-specific outputs. PR 1 establishes the
configuration, context, namespaces, and interfaces only; it does not fit a model or
write scientific outputs.

## Configuration

`pipeline_config.json` is the authoritative configuration format. The global input
dataset is inherited by every task unless a task supplies its own `input_dataset`:

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
              "smoothing_fwhm": 6.0,
              "high_pass": 0.01,
              "drift_model": "cosine",
              "hrf_model": "spm",
              "conditions": ["oneback", "twoback"],
              "confounds": {
                "motion": true,
                "motion_derivatives": true,
                "framewise_displacement": true,
                "acompcor": 5
              },
              "atlases": ["schaefer200", "schaefer400"]
            }
          },
          "analyses": {
            "activation": {
              "enabled": true,
              "model": "canonical_glm",
              "contrasts": {
                "2back_gt_1back": "twoback - oneback"
              }
            }
          }
        }
      }
    }
  }
}
```

Configured tasks default to `enabled: true`; specify `enabled: false` to temporarily
exclude a task. Configured analyses also default to enabled. Every enabled analysis
must reference a model declared by the same task.

The input dataset object contains a user-facing name and a path. For example, a task
can override the global fMRIPrep input with:

```json
"input_dataset": {
  "name": "afni",
  "path": "derivatives/afni"
}
```

Paths are resolved relative to the configuration/study root and are not copied into
the analysis output tree.

## Derivative namespaces

Model and analysis outputs have different namespaces, even when their names are the
same. Run-level contexts receive a run-specific directory so independent runs cannot
overwrite one another:

```text
derivatives/hbicproc/
  sub-001/
    ses-01/
      func/
        task-nback/
          run-1/
            models/
              canonical_glm/
            analyses/
              activation/
```

When no run entity exists, the `run-*` directory is omitted. The path abstraction is
implemented by `DerivativePathBuilder`; model and analysis implementations should use
it rather than constructing paths directly.

## Interfaces

- `TaskRunContext` identifies a subject/task/session/run and its discovered inputs.
- `ModelSpec` describes a model instance and its model-only fingerprint configuration.
- `AnalysisSpec` describes an analysis, its model reference, and required derivatives.
- `TaskPlan` binds task configuration to its input dataset and model/analysis specs.

Atlas configuration is deliberately excluded from model fingerprint configuration.
Later atlas derivatives can therefore be regenerated without refitting a GLM.

## CLI behavior

The existing command names remain unchanged:

```text
hbicproc analysis sub-001
hbicproc analysis --all
```

The PR 1 analysis service validates configuration and reports the task/model/analysis
plan. It intentionally performs no model fitting. Model execution, dependency reuse,
atlas generation, and activation outputs are introduced by later PRs.
