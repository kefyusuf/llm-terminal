# Codebase Structure

**Baseline date:** 2026-09-07  
**Baseline revision:** `f371cbf357731db729345d1cd29bc663bfb6edf7`

This document describes the current repository layout. It replaces the older mapper output that referenced a root `app.py` monolith and outdated file/test counts.

Continuation reviewed on 2026-10-04 against unmerged source `6bab5a6` through
PR #140. The original baseline above is historical; exact delivery/evidence
state is maintained in [continuation](../continuation.md).

## Top-Level Layout

```text
llm-terminal/
├── main.py                  # Primary TUI entry point
├── tui_app.py               # Base Textual application implementation
├── cli.py                   # Click CLI
├── api_server.py            # Local REST API
├── config.py                # Pydantic settings
├── pyproject.toml           # Packaging/tooling config
│
├── app/                     # Runtime TUI support modules
│   ├── viewer.py            # Runtime AIModelViewer subclass/provider selector
│   ├── modals.py            # Detail/download/plan/comparison modals
│   ├── widgets.py           # Focused Textual widgets
│   ├── download_manager.py  # TUI-side download coordination
│   ├── search_results_state.py
│   └── search_constants.py
│
├── core/                    # Domain logic and infrastructure helpers
├── providers/               # Search/runtime providers + capabilities
├── search/                  # Search orchestration, cache, pagination rules
├── results/                 # Result layout/render/filter helpers
├── downloads/               # Background download service + client/store/runner
├── terminal_ui/             # Theme/style assets and isolated legacy internals
├── scripts/                 # Dev/release/maintenance commands
├── requirements/            # Intent files + committed platform locks
├── tests/                   # Unit, contract, smoke and optional live tests
├── docs/                    # Maintained developer/product documentation
└── .planning/               # Active roadmap + current codebase baseline docs
```

## Entry Points

### `main.py`

Installed script: `ai-model-explorer`

Creates `app.viewer.AIModelViewer`. `app/viewer.py` subclasses the base `AIModelViewer` from `tui_app.py` and owns runtime provider-selector behavior.

### `cli.py`

Installed script: `ai-model-explorer-cli`

Commands include system information, search, fit, recommend, plan, scores,
offline saved-search compare, doctor and cache operations. Search/plan have
versioned JSON exports; saved runtime facts can augment planning.

### `api_server.py`

Invoked with `python -m api_server`.

Provides local machine-readable endpoints on port 8787 by default.

### `downloads/download_service.py`

Auto-started by the client when required. Runs the persistent background download queue/service on port 8765 by default.

## `app/` — TUI Composition

Use this package for presentation responsibilities that can be kept out of the base Textual application.

- `viewer.py` — provider selector integration and runtime viewer extension.
- `modals.py` — modal screens.
- `widgets.py` — reusable Textual widgets.
- `download_manager.py` — client-side download coordination/poll application.
- `search_results_state.py` — search-result state and invariants.
- `search_constants.py` — filter/sort/use-case UI constants and compact tags.

When adding a new modal or provider-selector behavior, prefer `app/` over expanding `tui_app.py` unless the behavior fundamentally belongs to the base application event loop/layout.

## `search/` — Search Pipeline

Key files:

- `search_orchestrator.py` — parallel provider fan-out/fan-in, cancellation, structured diagnostic aggregation.
- `search_orchestration.py` — provider selection, capability-aware pagination and query/status helpers.
- `search_cache.py` — in-memory search cache with hardware-aware invalidation and stale-entry access.

New provider orchestration behavior should normally be implemented here rather than directly in Textual callbacks.

## `providers/` — Provider Backends

Key files:

- `base.py` — `SearchResult` contract.
- `capabilities.py` — canonical provider capability metadata.
- `__init__.py` — `BaseProvider` interface + registry/detection helpers.
- `ollama_provider.py` — Ollama local API + remote registry HTML discovery.
- `hf_provider.py` — Hugging Face search/metadata integration.
- `lmstudio_provider.py` — LM Studio local API.
- `docker_provider.py` — Docker Model Runner local API.
- `mlx_provider.py` — Apple Silicon/local cache discovery.

When adding a provider:

1. implement the provider module/interface,
2. declare canonical capabilities,
3. register discovery/detection,
4. add focused provider contract/error tests,
5. expose it to TUI/CLI/REST only when that surface intentionally supports it.

Do not infer capabilities from display labels or result length.

## `core/` — Shared Domain Logic

Contains:

- hardware detection,
- scoring and GPU bandwidth data,
- model/MoE/quantization intelligence,
- SQLite metadata cache,
- shared HTTP session/retry helpers,
- structured provider error types,
- logging and utility functions.

`artifacts.py` validates public pinned artifact declarations; `exports.py`
owns bounded saved-result contracts; `model_facts.py` owns filtered runtime
facts and explicitly assumed memory scenarios. These do not change scoring.

`core/` should remain independent of Textual UI details.

## `results/` — View Logic

Contains pure or mostly-pure helpers for:

- responsive columns,
- cell presentation,
- text truncation/alignment,
- filtering/sorting/result identity.

Use these modules for deterministic presentation logic that can be tested without starting the TUI.

## `downloads/` — Download Service

The package is split between client/API/state/runner concerns instead of one service monolith.

Important modules include:

- `download_service.py` — service composition/startup and worker dispatch,
- `api.py` — service HTTP API handling,
- `store.py` — persistent download-job SQLite store,
- `runner.py` — job execution/process handling,
- `preflight.py` — authoritative HF path/disk plans and byte/digest verification,
- `parent_lifetime.py` — EOF guard for the service-owned HF child,
- `service_client.py` — TUI/client requests and service compatibility handling,
- `download_history.py`, `download_lifecycle.py`, `download_status.py` — reusable state helpers,
- `download_manager.py` — command/payload helpers,
- `hf_downloader.py` — focused Hugging Face downloader entry path.

The worker pool is bounded and configurable; download state is persisted independently of the TUI process.

Relevant qualification tools include `scripts/package_candidate.py` (one
manifest/wheel/sdist input), `verify_installed_package.py` (installed origin and
entrypoints), `verify_hf_recovery.py` (opt-in pinned sample),
`verify_sqlite_compatibility.py` (read-only candidate backup compatibility),
`benchmark_results_table.py` (bounded headless refresh costs) and
`calibrate_ollama.py` (prepared local runtime only). Operational evidence is
linked from `docs/release-readiness.md`.

## `tests/` — Verification

The verify lane contains more than 600 tests as of this baseline. Tests cover:

- core scoring/model intelligence/hardware helpers,
- provider parsing/retry/structured error contracts,
- search orchestration/cancellation/pagination,
- cache concurrency,
- REST/CLI contracts,
- download service/store/lifecycle,
- TUI state/view helpers and smoke paths,
- dev/release scripts.

Live external integration tests are opt-in (`--run-live`).

## Runtime Data

Runtime data is not assumed to live under a repository-local `data/` directory. Defaults use OS-specific per-user application data paths. Configuration variables can override cache DB, download DB, and model destinations.

Large model files, runtime DBs, logs, virtualenvs and generated state must stay out of version control.

## Where to Add New Work

| Change | Preferred location |
|---|---|
| Provider implementation | `providers/{name}_provider.py` + capabilities/registry |
| Provider fan-out/cancel/error aggregation | `search/search_orchestrator.py` |
| Pagination/provider selection rules | `search/search_orchestration.py` |
| Search caching | `search/search_cache.py` |
| Scoring/model intelligence | `core/` |
| Result sorting/formatting/layout | `results/` |
| New modal/widget | `app/modals.py` / `app/widgets.py` |
| TUI download behavior | `app/download_manager.py` + `downloads/` helpers |
| Background download execution | `downloads/` service/store/runner modules |
| CLI behavior | `cli.py` |
| REST behavior | `api_server.py` |
| Documentation/process | `README.md`, `docs/`, `.planning/` |

## Documentation Rule

Structural changes require a review of this file and `ARCHITECTURE.md` before the implementation is considered fully documented. See `docs/maintenance.md`.
