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