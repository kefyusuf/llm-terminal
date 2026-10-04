# Repository Structure

## Root Principles

- Keep root focused on entrypoints and project metadata.
- Group domain modules into packages once a flat prefix family forms.
- Store runtime databases, logs, and downloaded models in OS-specific per-user application-data directories by default; `config.py` exposes path overrides.
- Store dependency intent and lock files under `requirements/`.
- Store implementation notes and release artifacts under `docs/`.

## Current Layout

```text
llm-terminal/
  api_server.py
  app/
  cli.py
  config.py
  core/
  main.py
  pyproject.toml
  README.md
  docs/
  downloads/
  providers/
  requirements/
  results/
  scripts/
  search/
  terminal_ui/
  tests/
  tui_app.py
  .planning/
```

## Package Boundaries

- `app/`: runtime viewer extensions, responsive startup, modals, widgets, search state, and TUI-side download coordination.
- `downloads/`: download state, lifecycle helpers, command builder, and background service.
  Includes the local service client used by the TUI and CLI, plus the standalone HF downloader.
- `core/`: shared cache persistence, hardware detection, logging, model metadata, scoring, quantization, and parsing helpers.
- `results/`: results-table layout, formatting, and filtering.
- `search/`: in-memory search cache, provider-aware pagination helpers, and parallel search orchestration with cancellation and diagnostics.
- `scripts/`: developer, release, and maintenance entry scripts (Python and batch wrappers).
- `terminal_ui/`: theme/style assets and isolated legacy UI internals.
- `providers/`: external provider integrations.

## Runtime Entry Paths

- TUI: `main.py` -> `app.viewer.AIModelViewer` -> `app.startup_viewer.AIModelViewer` -> `tui_app.AIModelViewer` -> Textual `App`.
- CLI: `cli.py` calls shared hardware/scoring services and provider search functions directly; it does not route every search through the TUI orchestrator.
- REST: `api_server.py` owns its HTTP handlers and currently searches Ollama and Hugging Face.
- Downloads: `python -m downloads.download_service` starts a separate localhost service; `downloads/api.py`, `downloads/store.py`, and `downloads/runner.py` own transport, persistence, and execution respectively.

The runtime viewer adds asynchronous service startup, provider selectors, live-first stale-cache fallback, and table-content consistency checks over the base TUI. Inspect this inheritance chain before changing a method in `tui_app.py`: the runtime may override it.

Search results use `providers.base.SearchResult` and model dictionaries described by `core.models.ModelResult`. The orchestrator returns `SearchOutcome`; the TUI applies it to `app/search_results_state.py` and renders it through `results/` helpers. Provider capabilities, rather than result counts, govern pagination and download support.

There are two distinct caches: `search/search_cache.py` holds in-process search pages; `core/cache_db.py` persists metadata and hardware snapshots in SQLite. The stale search fallback is limited to retained entries in the current process.

Download status updates can use `DataTable.update_cell()` when ordered model content and column layout stay unchanged. The runtime viewer invalidates that fast path when rendered content changes; structural changes still rebuild rows.

## Next Refactor Candidates

- Re-evaluate the remaining root-level shared modules only after a clearer ownership boundary emerges.
