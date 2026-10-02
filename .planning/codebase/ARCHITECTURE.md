# Architecture

**Baseline date:** 2026-09-07  
**Baseline revision:** `f371cbf357731db729345d1cd29bc663bfb6edf7`

## Overview

AI Model Explorer is a terminal-first local application with three user-facing surfaces over shared core logic:

- Textual TUI (`main.py` → `app.viewer.AIModelViewer`),
- Click CLI (`cli.py`),
- localhost REST API (`api_server.py`).

Model discovery is provider-based. Download execution is isolated in a separate localhost HTTP service so jobs can continue independently of TUI lifecycle.

The architecture is no longer the old single-file `app.py` monolith described by the 2026-06-09 mapper output. The main Textual implementation still lives in `tui_app.py`, but provider selection, modals, widgets, search state, and download-management concerns have been extracted into focused modules under `app/`; search fan-out is isolated in `search/search_orchestrator.py`.

## Major Components

### 1. Runtime TUI

Entry path:

```text
main.py
  -> app/viewer.py
       -> tui_app.AIModelViewer
```

Responsibilities:

- Textual layout and event handling,
- user search/filter/sort actions,
- cache/stale-cache use,
- result rendering,
- hardware status presentation,
- coordination with download manager and search orchestrator.

Supporting modules under `app/`:

- `viewer.py` — runtime subclass/provider-selector integration,
- `modals.py` — model detail, download, plan, and comparison modal screens,
- `widgets.py` — focused Textual widgets,
- `download_manager.py` — TUI-side download coordination,
- `search_results_state.py` — search-result state/invariants,
- `search_constants.py` — filter/sort/use-case UI constants.

`tui_app.py` remains the largest presentation module and is still a performance/refactor sensitivity area, but it is no longer responsible for every concern in the application.

### 2. Search Layer

Key modules:

- `search/search_orchestrator.py` — parallel provider fan-out/fan-in, cancellation polling, deterministic grouping, structured diagnostics,
- `search/search_orchestration.py` — provider selection, pagination/capability rules, query/status helpers,
- `search/search_cache.py` — in-memory hardware-aware search cache with stale-entry access for TUI offline fallback.

`SearchOrchestrator` uses a bounded `ThreadPoolExecutor` and keeps provider result grouping deterministic:

1. Ollama,
2. Hugging Face,
3. extra providers.

Cancellation is polled while futures are pending so the caller can return without waiting for still-running provider workers.

### 3. Provider Layer

Providers:

- Ollama,
- Hugging Face,
- LM Studio,
- Docker Model Runner,
- MLX.

Provider capabilities are defined centrally in `providers/capabilities.py`. Capability metadata is used to decide searchability, pagination, installed-list support, downloadability, and provider-selection behavior.

The provider contract uses `SearchResult` (`providers/base.py`) with:

- `results`,
- legacy human-readable `errors`,
- `has_more_pages`,
- machine-readable `structured_errors` (`ProviderError`).

Transient/search/parse/I/O failures should normally be contained inside provider results instead of escaping as uncaught exceptions.

### 4. Error and Diagnostic Flow

`core/errors.py` defines `ProviderError` metadata such as:

- provider slug,
- stable code,
- message,
- retryability,
- optional HTTP status,
- optional retry-after duration.

Diagnostic flow:

```text
provider
  -> SearchResult.errors + SearchResult.structured_errors
  -> SearchOrchestrator SearchOutcome
  -> TUI status/error handling
```

Other surfaces preserve diagnostics independently:

- REST `/api/v1/models` returns additive `errors` and `structured_errors` arrays while preserving HTTP-200 partial-result semantics.
- CLI `search`, `fit`, and `recommend` emit provider warnings on stderr. `recommend --json` keeps stdout as the existing JSON array contract.

The remaining goal is not to remove all broad catches, but to ensure user-impacting failures are observable and correctly classified.

### 5. HTTP Client Layer

`core/http_client.py` owns the shared `requests.Session` used by requests-based providers. It provides:

- connection reuse,
- retry/backoff for retryable GET failures,
- handling for 429 and common 5xx responses.

Hugging Face also uses `huggingface_hub` APIs for search/download-specific operations.

### 6. Core Domain

`core/` contains UI-independent domain services:

- hardware detection and snapshots,
- scoring and GPU-bandwidth lookup,
- MoE/model-size/quantization intelligence,
- metadata/hardware SQLite cache,
- shared utilities,
- logging,
- typed provider errors.

`core/cache_db.py` uses a shared SQLite connection guarded by an `RLock`, reopens when the configured path changes, and retries after SQLite connection errors.

### 7. Results Layer

`results/` contains mostly pure presentation helpers:

- responsive column layout,
- cell markup,
- text alignment/truncation,
- filtering/sorting/view selection.

The main TUI still performs full DataTable rebuilds in important refresh paths. Incremental updates are a planned performance improvement, not a correctness requirement.

### 8. Download Layer

The download system has two sides:

**TUI/client side**

- `app/download_manager.py`,
- `downloads/service_client.py`,
- lifecycle/history/status helpers.

**Background service side**

- `downloads/download_service.py`,
- `downloads/api.py`,
- `downloads/store.py`,
- `downloads/runner.py`.

The service runs on loopback (default `127.0.0.1:8765`) and persists jobs in SQLite. Work is dispatched through a bounded `ThreadPoolExecutor`; default concurrency is 2 and is configurable with `AIMODEL_DOWNLOAD_MAX_WORKERS`.

The service supports an optional bearer token for non-health endpoints. Non-loopback bind/client hosts are rejected until authenticated TLS transport is implemented.

### 9. REST API

`api_server.py` exposes localhost programmatic access on `127.0.0.1:8787` by default.

Current model-search providers exposed through `/api/v1/models` are Ollama and Hugging Face. The provider descriptor endpoint exposes the canonical provider registry and identifies which providers are supported by the REST model endpoint.

Request validation returns 400 for invalid provider/limit/context/sort inputs rather than relying on outer 500 handling.

### 10. CLI

`cli.py` provides:

- hardware/system info,
- model search,
- hardware-fit listing,
- recommendations,
- hardware planning,
- scoring,
- cache helpers.

Provider errors are surfaced on stderr. JSON recommendation output remains stdout-only JSON for scripting compatibility.

## Primary Search Flow

```text
query/filter action
  -> build provider selection + query key
  -> SearchCache lookup
       -> fresh hit: use cached result
       -> stale hit may be used as TUI offline fallback
  -> background SearchOrchestrator
       -> providers run in parallel
       -> SearchResult diagnostics collected
       -> cancellation observed between future completions
  -> SearchOutcome
  -> cache/update state
  -> filter/sort/render DataTable
```

Only the selected single paginated provider may authorize a next page. Multi-provider search does not synthesize continuation from result counts.

## Primary Download Flow

```text
TUI action
  -> service_client HTTP request
  -> download-service API
  -> DownloadStore SQLite state
  -> bounded worker pool
  -> runner (Ollama pull or HF download path)
  -> persisted state/progress
  -> TUI polling snapshot
  -> row/history update
```

## State and Storage

- Search result state: `SearchResultsState` + in-memory `SearchCache`.
- Persistent model/hardware metadata: SQLite cache DB.
- Download job state/history: download-service SQLite DB.
- Default DB/model directories: OS-specific user-data paths via configuration/platformdirs.
- Runtime configuration: singleton Pydantic Settings object (`config.settings`).

## Cross-Cutting Rules

The TUI owns a lazily initialized hardware monitor protected by a per-viewer lock. Metrics, search and model-detail workers reuse the completed monitor; UI-side cache lookup never triggers detection. Cached-header rendering leaves Ollama status pending until worker polling completes. Failed initialization remains retryable and expected search/detail probe failures are applied as UI diagnostics. See [hardware startup](../../docs/hardware-startup.md).

Provider discovery has surface-specific scheduling. The runtime TUI starts with the canonical Ollama/Hugging Face selectors and probes optional providers in a guarded Textual thread worker after mount, refreshing every 30 seconds. REST owns one `providers.discovery.ProviderDiscovery` snapshot per server; handlers return pending/last-known values without waiting for network detection. Freshness, refresh-in-progress and sanitized refresh-error metadata are additive. The direct registry detection helpers remain synchronous for callers that explicitly request a completed probe. See [provider discovery](../../docs/provider-discovery.md).

- Public behavior is shared conceptually, but TUI/CLI/REST are not forced through one abstraction when that would break surface-specific contracts.
- Provider capabilities are authoritative; UI heuristics should not invent capabilities.
- Diagnostic metadata is additive and must not require string parsing for machine-readable callers.
- Correctness fixes should preserve partial-success behavior when one provider fails and another succeeds.
- Documentation must be re-reviewed after architecture-affecting merges; see `docs/maintenance.md`.
