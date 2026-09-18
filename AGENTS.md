# AGENTS.md

hbicproc is organized as independent pipeline stages.

When implementing a feature:

1. Work only within the requested stage.
2. Inspect only files relevant to that stage.
3. Avoid cross-stage refactoring.
4. Treat upstream outputs as stable interfaces.
5. Reuse existing patterns before introducing new architecture.
6. Implement reports incrementally.
7. Keep QC measurements separate from analysis decisions.

Prefer small, localized changes over project-wide redesign.

## Logging

HBICPROC uses Python logging for all operational messages.

- Configure logging only at application startup.
- Modules must obtain loggers via:

  logger = logging.getLogger(__name__)

- Do not call logging.basicConfig() from library modules.
- Prefer logger calls over print() for status reporting, diagnostics, warnings, and errors.

Log level conventions:

- DEBUG: developer diagnostics and internal state useful for troubleshooting.
- INFO: high-level pipeline progress, selected configuration, manifests, and subject/session/run counts.
- WARNING: unexpected conditions where processing can continue.
- ERROR: failures that prevent completion of the current operation.

Guidelines:

- Log the start and completion of major pipeline stages.
- Log population counts before and after filtering or exclusions.
- Avoid excessive INFO logging inside large loops.
- Use DEBUG for detailed per-subject or per-file information.
- Logging should help explain what the pipeline did and why.
- Do not use print() for operational messages.
- Use logger.debug(), logger.info(), logger.warning(), or logger.error().
- Logging format (timestamp, level, module name) is supplied centrally by the logging configuration.
- Log messages should contain only the event being reported.

Examples:

  logger.info("Loaded decision manifest '%s'", manifest.analysis_id)
  logger.info("Subjects before exclusions: %d", len(subjects))
  logger.info("Subjects after exclusions: %d", len(filtered_subjects))
  logger.warning("fMRIPrep mask unavailable; using Nilearn fallback")