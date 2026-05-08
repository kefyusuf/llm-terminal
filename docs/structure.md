# Repository Structure

## Root Principles

- Keep root focused on entrypoints and project metadata.
- Group domain modules into packages once a flat prefix family forms.
- Store runtime artifacts under `data/`.
- Store dependency intent and lock files under `requirements/`.
- Store implementation notes and release artifacts under `docs/`.

## Current Layout

```text
llm-terminal/
  api_server.py
  app.py
  cli.py
  config.py
  core/
  main.py
  pyproject.toml
  README.md
  data/
  docs/
  downloads/
  models/
  providers/
  requirements/
  results/
  scripts/
  search/
  terminal_ui/
  tests/
```

## Package Boundaries

- `downloads/`: download state, lifecycle helpers, command builder, and background service.
  Includes the local service client used by the TUI and CLI, plus the standalone HF downloader.
- `core/`: shared cache persistence, hardware detection, logging, model metadata, scoring, quantization, and parsing helpers.
- `results/`: results-table layout, formatting, and filtering.
- `search/`: in-memory search cache and provider pagination helpers.
- `scripts/`: developer, release, and maintenance entry scripts (Python and batch wrappers).
- `terminal_ui/`: Textual-specific modules, stylesheets, and runtime theme definitions.
- `providers/`: external provider integrations.

## Next Refactor Candidates

- Re-evaluate the remaining root-level shared modules only after a clearer ownership boundary emerges.