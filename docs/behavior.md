# Behavioral events processing

The behavioral events subsystem lives at `pipeline/processing/behavior/` and is wired into the pipeline
as the `behavior` stage (`pipeline/stages/behavior.py`), following the same stage/processing-service
convention as every other stage.

## Overview

- Read raw E-Prime .edat3 files through the EPrimeReader abstraction.
- Keep task-specific parsing in plugin parsers under `pipeline/processing/behavior/parsers`.
- Use JSON task configuration files for column mapping and optional metadata.
- Generate BIDS-compatible events.tsv and events.json outputs.

## Adding a new task

1. Create a new task configuration JSON file in the configured behavior task config directory (default: code/behavior).
2. Implement a new parser class that inherits TaskParser and register it in the parser registry.
3. Add the task name to the parser registry in `pipeline/processing/behavior/service.py` or add a registration call for the new class.
4. Run the CLI command:

   hbicproc behavior sub-001 --task your-task

## Notes

- Parsers should only transform raw E-Prime data into events tables; they should not write files.
- The writer layer is responsible for generating TSV and JSON sidecar outputs.
- The validator layer enforces the required BIDS-style columns before export.
