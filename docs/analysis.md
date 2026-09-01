# Subject-level analysis stage

The analysis feature is implemented as a normal pipeline stage rather than as a standalone CLI bypass. The stage wrapper sits in the package’s usual layering:

- CLI -> runner -> stage -> processing service -> plugin registry -> plugin implementations

## Stage contract

- `pipeline/stages/analysis.py` defines `AnalysisStage`.
- `pipeline/processing/analysis/service.py` contains the public `run(subject, config, dry_run=False, rerun=False)` entrypoint used by the stage.
- The stage is registered in `pipeline/stages/__init__.py` so it is auto-exposed by the generic `STAGE_CLASSES` CLI wiring.
- Since analysis is an optional subject-level workflow, it is intentionally not included in `PipelineRunner.stage_order` and therefore requires an explicit subject or `--all` when invoked.

## Plugin registry

The plugin layer remains available behind the processing service for extensibility:

- `pipeline/processing/analysis/base.py` defines the plugin interface.
- `pipeline/processing/analysis/registry.py` discovers and runs plugin implementations.
- `pipeline/processing/analysis/activation/plugin.py` is the first concrete plugin.
- `pipeline/processing/analysis/activation/derivatives.py` builds BIDS-derivative output names consistently.

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

- effect variance is generated with `glm.compute_contrast(..., output_type="effect_variance")`
- design matrices are plotted with `nilearn.plotting.plot_design_matrix`
- residuals are read from `glm.residuals_` when available, with a warning and no failure if residual images are not exposed by the current Nilearn implementation
- output images use effect-map terminology instead of beta-map terminology

### Output naming

Output files are written as BIDS-style derivative names with entity labels such as `sub-*`, `ses-*`, `task-*`, `run-*`, `desc-*`, and `stat-*`. Condition outputs use effect-map naming and contrast variance outputs use Nilearn effect-variance images.

## Future analyses

New analysis methods should follow the same pattern: add a plugin class under `pipeline/processing/analysis/...`, keep it isolated from the stage wrapper, and expose it through the `analysis` stage so it integrates with the existing pipeline conventions.
