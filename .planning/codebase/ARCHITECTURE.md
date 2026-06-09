# Architecture

**Analysis Date:** 2026-06-09

## Pattern Overview

**Overall:** Monolithic TUI application with layering via internal packages + background subprocess service

The application runs as a single Textual-based TUI process (`app.py:673`, class `AIModelViewer`) that orchestrates the entire user experience. A separate background process (`downloads/download_service.py`) handles model downloads asynchronously, communicating over HTTP on `127.0.0.1:8765`.

**Key Characteristics:**
- Provider plugin model via `BaseProvider` ABC (`providers/__init__.py:16`) — Ollama, HuggingFace, LM Studio, Docker, MLX
- 4-dimension scoring engine (`core/scoring.py`) — Quality, Speed, Fit, Context metrics
- Dual-mode UI: "comfortable" (full) and "compact" (minimal) views
- Dual-cache strategy: SQLite for persistent metadata + in-memory for search results
- CLI alternative via Click (`cli.py`) and REST API via `http.server` (`api_server.py:8787`)

## Layers

**1. TUI Presentation (Heaviest Layer):**
- Purpose: Render the terminal UI, handle keyboard/mouse input, manage widgets
- Location: `app.py` (2466 lines — the monolith)
- Contains: `AIModelViewer(App)` main class, 4 modal screens (`ModelDetailModal`, `DownloadJobModal`, `PlanModeModal`, `ComparisonModal`), `SystemInfoWidget`
- Depends on: All other layers (core, providers, downloads, search, results)
- Used by: `main.py` entry point

**2. CLI Layer:**
- Purpose: Command-line access to search, scoring, hardware info, cache management
- Location: `cli.py` (398 lines)
- Contains: 8 Click commands (`info`, `system`, `search`, `fit`, `recommend`, `plan`, `scores`, `cache-clear`, `cache-stats`, `version`)
- Depends on: `core`, `providers`
- Used by: `pyproject.toml` entry point `ai-model-explorer-cli = "cli:cli"`

**3. REST API Layer:**
- Purpose: Machine-readable HTTP API for programmatic access
- Location: `api_server.py` (339 lines)
- Contains: `ModelAPIHandler`, `create_server()`, `run_server()`, `start_server_background()`
- Depends on: `core`, `providers`
- Endpoints: `/health`, `/api/v1/system`, `/api/v1/models`, `/api/v1/models/{name}/plan`, `/api/v1/scores/{name}`, `/api/v1/providers`

**4. Core Domain:**
- Purpose: Business logic — hardware detection, model scoring, caching, utilities
- Location: `core/` (1627 total lines across 8 files)
- Sub-modules:
  - `hardware.py` — `HardwareMonitor`, GPU vendor detection, Ollama process check (389 lines)
  - `scoring.py` — 4-dimension scoring engine, GPU bandwidth lookup, use-case weights (475 lines)
  - `model_intelligence.py` — MoE detection, quantization selection, size estimation (334 lines)
  - `models.py` — `ModelResult` TypedDict type definition (54 lines)
  - `utils.py` — Model name parsing, fit calculation, helpers (183 lines)
  - `cache_db.py` — SQLite-backed cache for model metadata + hardware snapshots (158 lines)
  - `logging_.py` — loguru setup with file rotation (33 lines)
- Used by: All other layers

**5. Provider Layer:**
- Purpose: Abstracted search backends for different model sources
- Location: `providers/` (1162 total lines across 6 files)
- Contains:
  - `__init__.py` — `BaseProvider` ABC, provider registry with lazy imports (154 lines)
  - `hf_provider.py` — HuggingFace Hub search + metadata enrichment (272 lines)
  - `ollama_provider.py` — Ollama registry scraping + local API (325 lines)
  - `lmstudio_provider.py` — LM Studio local API search (123 lines)
  - `docker_provider.py` — Docker Model Runner search (139 lines)
  - `mlx_provider.py` — Apple Silicon MLX local scan (149 lines)

**6. Search Layer:**
- Purpose: Search orchestration, caching, query building
- Location: `search/` (212 total lines across 2 files)
- Contains:
  - `search_orchestration.py` — Provider-aware pagination, query key building, status messages (123 lines)
  - `search_cache.py` — In-memory cache with hardware-aware invalidation (89 lines)

**7. Results Layer:**
- Purpose: Table layout, cell formatting, filtering/sorting
- Location: `results/` (589 total lines across 4 files)
- Contains:
  - `results_layout.py` — Responsive column computation for DataTable (167 lines)
  - `results_presenter.py` — Color-coded cell markup for all columns (241 lines)
  - `results_text.py` — Text truncation, alignment, header formatting (62 lines)
  - `results_view.py` — Filter and sort logic for model results (119 lines)

**8. Downloads Layer:**
- Purpose: Manage model downloads asynchronously
- Location: `downloads/` (1376 total lines across 8 files)
- This layer has TWO components:
  - **In-process client code:** `download_history.py`, `download_lifecycle.py`, `download_manager.py`, `download_status.py`, `service_client.py`, `hf_downloader.py`
  - **Background service:** `download_service.py` (796 lines) — Separate subprocess with its own HTTP server

**9. Configuration:**
- Purpose: Centralized settings with env var overrides
- Location: `config.py` (62 lines)
- Contains: `Settings` dataclass via `pydantic_settings.BaseSettings`

## Data Flow

**Primary Flow — Search:**

1. User types query in Input widget -> `on_input_submitted` (`app.py:1582`)
2. `start_search()` debounces (120ms), builds cache key -> `_dispatch_debounced_search()` (`app.py:1613`)
3. Checks `SearchCache` (in-memory, hardware-aware) — if hit, returns cached results
4. Cache miss -> `run_search_worker()` in background thread (`app.py:2204`)
5. Worker queries configured providers (Ollama, HF, LM Studio, Docker, MLX)
6. Each result is scored via `enrich_result_with_scores()` from `core/scoring.py`
7. Results combined -> cached -> `on_search_completed()` -> `refresh_table()`
8. `filter_results_for_view()` applies provider/use-case/hidden-gem/fit/sort filters (`app.py:2378`)

**Secondary Flow — Model Download:**

1. User clicks "Download" in `ModelDetailModal` -> `start_model_download()` (`app.py:1890`)
2. `service_client.create_job()` sends HTTP POST to download service on port 8765
3. Download service receives job -> `DownloadStore.upsert_job()` writes to SQLite queue
4. Worker loop picks up queued job -> spawns `subprocess.Popen` (ollama pull or HF snapshot_download)
5. `_run_stream_download_job()` / `_run_hf_api_download_job()` monitor progress via stdout parsing
6. Status polling: `app.py` polls `GET /jobs` every 1.5s via `_run_download_poll_worker()` (thread)
7. Results table updates download state column in real-time

**Tertiary Flow — Hardware Monitoring:**

1. `on_mount()` starts polling timers (`app.py:1220`):
   - `system_metrics_timer`: every 3s -> `request_system_info_refresh()`
   - `download_status_timer`: every 1.5s -> `request_download_poll()`
2. `_run_system_info_refresh_worker()` (thread) -> `monitor.get_specs()` + `check_ollama_running()`
3. Results update `SystemInfoWidget` header and cached `latest_specs`
4. Hardware changes automatically invalidate search cache

**State Management:**
- UI state: Stored as instance attributes on `AIModelViewer` (no reactive state management beyond Textual's built-in)
- Search results: `self.all_results` — a flat list of `ModelResult` dicts
- Comparison set: `self.comparison_set` — list of up to 4 model dicts
- Download registry: `self.download_registry` — dict keyed by `target_id`
- Configuration: Singleton `config.settings` object (Pydantic Settings)
- Persistent state: SQLite databases (cache + download queue)

## Key Abstractions

**BaseProvider (`providers/__init__.py:16`):**
- Purpose: Pluggable search backend interface
- Methods: `detect()`, `search()`, `list_installed()`, `search_with_installed()`
- Implementations: Ollama, HuggingFace, LM Studio, Docker, MLX (only Ollama + HF are used in primary search path)

**Scores dataclass (`core/scoring.py:136`):**
- Purpose: Immutable container for 4-dimension + composite scores
- Fields: `quality`, `speed`, `fit`, `context`, `composite`, `estimated_tok_s`
- Computed by `score_model()` with use-case-weighted aggregation

**ModelResult TypedDict (`core/models.py:4`):**
- Purpose: Canonical shape for all model result dicts across providers
- All fields optional (`total=False`) — providers populate different subsets

**HardwareMonitor (`core/hardware.py:100`):**
- Purpose: Detect and snapshot local hardware
- Detects: NVIDIA (pynvml/nvidia_smi), AMD (rocm-smi), Apple Silicon (system_profiler), Intel Arc (lspci)
- Output: CPU name/cores, RAM total/free, VRAM total/free, GPU name, vendor, backend
- Graceful fallback at every level

**DownloadService / DownloadStore (`downloads/download_service.py`):**
- Purpose: Background HTTP service for asynchronous model downloads
- Threading: Worker thread + locking for SQLite access + process tracking

## Entry Points

| Entry Point | File | How Invoked | Purpose |
|-------------|------|-------------|---------|
| `main()` | `main.py:23` | `ai-model-explorer` console script | Starts Textual TUI |
| `cli()` | `cli.py:14` | `ai-model-explorer-cli` console script | CLI commands |
| `api_server.main()` | `api_server.py:326` | `python -m api_server` | REST API server (port 8787) |
| `download_service.main()` | `downloads/download_service.py:742` | Auto-launched by `ensure_service_running()` | Background download worker (port 8765) |

## Error Handling

**Strategy:** Defensive logging with broad exception catches. Most error recovery is silent (log or status bar message).

**Patterns:**
- Broad `except Exception: pass` — used extensively across the codebase (e.g., `app.py:967`, `app.py:1237`, `app.py:1318`, etc.)
- HTTP error recovery: `service_client.py` retries once with `ensure_service_running()` on 404
- Graceful degradation: Hardware detection fails -> returns empty specs; Search fails -> shows error in results table
- Status bar updates via `self.update_status()` are the primary user-facing error channel
- No formal error types — errors are strings passed through dicts (e.g., `(results, errors)` tuples)

## Cross-Cutting Concerns

**Logging:** loguru configured in `core/logging_.py` — stderr (INFO) + file rotation (DEBUG). Used sparsely across provider and service code.

**Validation:** Minimal — Pydantic validates config only. Search queries are raw strings. No schema validation for API endpoints.

**Threading:**
- Textual's `@work(thread=True)` decorator for non-blocking operations
- Manual `call_from_thread()` for cross-thread UI updates
- `threading.Lock` for SQLite in `cache_db.py` and `DownloadStore`
- `threading.Event` for graceful shutdown in download service

---

*Architecture analysis: 2026-06-09*
