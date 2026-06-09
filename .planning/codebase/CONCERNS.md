# Codebase Concerns

**Analysis Date:** 2026-06-09

## Tech Debt

### 1. `app.py` Monolith — Extreme Size (2466 LOC)
- **Issue:** The main TUI application in `app.py` is a massive single-file monolith containing the `AIModelViewer` App class plus 4 modal screen classes. It simultaneously manages: UI layout, keyboard bindings, search orchestration, download lifecycle, hardware polling, pagination, comparison logic, theme management, and status updates.
- **Files:** `app.py` (lines 1-2466)
- **Impact:**
  - Single-responsibility violations — `AIModelViewer` has ~50+ instance attributes and ~60+ methods
  - Extremely difficult to test — only 3 test files target TUI behavior
  - High cognitive load for any modification
  - Long methods like `run_search_worker` (~120 lines), `refresh_table` (~100 lines), `_apply_ui_mode` (~68 lines)
  - All four modals (`ModelDetailModal`, `DownloadJobModal`, `PlanModeModal`, `ComparisonModal`) are mixed into the same file
- **Fix approach:** Extract modal screens to `app/modals.py`, extract search orchestration from `AIModelViewer` to a `SearchCoordinator` delegate, extract download management to a `DownloadManager` delegate

### 2. Broad `except Exception: pass` Pattern
- **Issue:** The codebase uses silent exception swallowing as its primary error handling strategy. Errors in hardware polling, cache access, UI refresh, and search are silently dropped.
- **Files:** `app.py` (lines ~967, 1101, 1237, 1318, 1505, 1532, 1605, 2284, 2285), `core/cache_db.py` (pass on ~72, 86, 124, 144, 158), `downloads/download_service.py` (pass on ~573-574), and many more
- **Impact:** Silent failures hide real issues. A broken GPU detection, failed search, or stale cache produces no visible error — the UI just shows empty results or defaults.
- **Fix approach:** Replace with structured logging via loguru, add warning-level log statements, use specific exception types where possible

### 3. Duplicated Model-Size Estimation Logic
- **Issue:** Two separate functions estimate model size from name tokens: `estimate_model_size_gb()` in `core/utils.py:81` (11 branches, plain if/elif chain) and `estimate_model_size_gb_v2()` in `core/model_intelligence.py:213` (table-driven approach with `_PARAM_SIZE_MAP`). Both are used in different code paths.
- **Files:** `core/utils.py:81-110`, `core/model_intelligence.py:167-255`
- **Impact:** Inconsistent results. `utils.estimate_model_size_gb` uses a flat if/elif chain with hardcoded values; `model_intelligence.estimate_model_size_gb_v2` uses a dict table with MoE awareness. When a model name doesn't match, they return different results.
- **Fix approach:** Deprecate `utils.estimate_model_size_gb`, make `model_intelligence.estimate_model_size_gb_v2` the single source of truth, update all callers

### 4. HTML Screen-Scraping for Ollama Registry
- **Issue:** The Ollama provider scrapes `ollama.com/search` and `ollama.com/library/{name}` using BeautifulSoup with fragile HTML structure assumptions.
- **Files:** `providers/ollama_provider.py:70-127` (`_extract_models_table_rows`), `providers/ollama_provider.py:239-260` (search page anchor parsing)
- **Impact:** Brittle — any HTML structure change on ollama.com breaks search. No pagination support despite Ollama using htmx infinite scroll (currently ignored).
- **Fix approach:** Request an official Ollama search API, or use a more robust scraping strategy with structured data extraction

### 5. Import-Within-Function Pattern Throughout `app.py`
- **Issue:** `app.py` uses late/conditional imports scattered within method bodies (e.g., `from providers import get_provider_filter_labels` inside `action_cycle_provider`, `from core.model_intelligence import plan_hardware_for_model` inside `action_open_plan_mode`).
- **Files:** `app.py` (lines 1267, 1314, 1293, 2266)
- **Impact:** Unnecessary overhead on each action invocation, obscured dependency graph, can mask circular import issues
- **Fix approach:** Move all imports to module top level; resolve any circular dependency issues

### 6. No Retry/Rate-Limiting Logic on Provider HTTP Calls
- **Issue:** Provider search calls have no retry or exponential backoff. An HTTP 429 (rate limit) or transient failure immediately surfaces as an error to the user.
- **Files:** `providers/hf_provider.py:118-130`, `providers/ollama_provider.py:216-235`
- **Impact:** Rate-limited requests fail immediately even though the user could wait and retry. No caching of failed responses.
- **Fix approach:** Add `tenacity` or custom retry decorator with exponential backoff for provider HTTP calls

### 7. No Formal Error Types — Errors as Strings
- **Issue:** Errors are passed around as plain strings in `(results, errors)` tuples. There is no error hierarchy, no structured error metadata, and no machine-readable error codes.
- **Files:** All provider `search()` methods, `app.py` error handling paths
- **Impact:** Callers cannot differentiate error types (rate limit vs. network vs. auth vs. parse error) without string parsing
- **Fix approach:** Define a lightweight error class hierarchy (`RateLimitError`, `NetworkError`, `ParseError`, etc.) with codes and user-friendly messages

## Known Bugs

### 1. Smoke Mode Race Condition
- **Issue:** `_finish_smoke_mode()` in `app.py:1165` calls `self.exit()` which may be invoked before the app is fully mounted. The `set_timer(1, _finish_smoke_mode)` backup fires simultaneously with `call_after_refresh(self._finish_smoke_mode)`.
- **Files:** `app.py:1190-1193`
- **Trigger:** Running with `AIMODEL_SMOKE=1`
- **Workaround:** Not user-facing (dev-only smoke test mode)

### 2. ComparisonModal Duplicate Key Handling
- **Issue:** `ComparisonModal.compose()` iterates rows with accessor lambdas. If two models in the comparison set share the same `name` prefix, the `Strip_markup` call can produce zero-length text that breaks the pipeline rendering.
- **Files:** `app.py:627-655`
- **Trigger:** Selecting two models with identical names but different sources

### 3. Download History Table Sort Key Edge Case
- **Issue:** `refresh_download_history_table()` sorts by `created_at` then `target_id`. If `created_at` is missing from an entry, `str(entry.get("created_at", 0))` produces `"0"` for the sort, which may interleave incorrectly with entries that have real timestamps.
- **Files:** `app.py:1990-1997`
- **Trigger:** Download entries where `created_at` field was not set during upsert

## Security Considerations

### 1. HuggingFace Token Leaked to Subprocess Environment
- **Issue:** The download service passes the HuggingFace token to subprocesses via environment variables: `env["HF_TOKEN"]` and `env["AIMODEL_HF_TOKEN"]` (in `downloads/download_service.py:348-349`). Any child process inherits these.
- **Files:** `downloads/download_service.py:344-349`
- **Risk:** If a subprocess is compromised or leaks, the HF token is exposed
- **Current mitigation:** Subprocess runs locally only
- **Recommendations:** Use stdin pipe instead of env vars for token passing; or scope token permissions to read-only

### 2. No Input Validation on API Server
- **Issue:** `api_server.py` takes raw query parameters (`search`, `provider`, `limit`, etc.) without validation. Integer casts (`int(params.get("limit", ["20"])[0])`) can raise `ValueError` on malformed input.
- **Files:** `api_server.py:118-124`
- **Risk:** Malformed requests cause 500 errors instead of graceful 400 responses
- **Recommendations:** Add parameter validation with try/except, return structured 400 errors

### 3. Subprocess Shell Execution
- **Issue:** `terminal_ui/app.py:106-124` (`run_shell_command`) uses `shell=True` with user-provided input (command text from the Input widget). While `run_shell_command` receives already-typed commands, the `subprocess.run(cmd, shell=True)` pattern is dangerous for any untrusted input.
- **Files:** `terminal_ui/app.py:109-110`
- **Risk:** If this code path is ever exposed to external input, shell injection is trivial
- **Current mitigation:** Only runs in the legacy `ObsidianConsole` which isn't the primary entry point
- **Recommendations:** Replace `shell=True` with argument list in the legacy widget, or document as deprecated

## Performance Bottlenecks

### 1. Synchronous HTTP Calls in Provider Search
- **Issue:** Search across multiple providers (Ollama + HF + LM Studio + Docker + MLX) runs sequentially, not in parallel. Each provider's HTTP call blocks the search worker thread.
- **Files:** `app.py:2204-2291` (`run_search_worker`)
- **Cause:** Sequential provider search. Ollama finishes first, then HF starts.
- **Improvement path:** Use `concurrent.futures.ThreadPoolExecutor` to run provider searches in parallel with a timeout

### 2. GPU Bandwidth Linear Search
- **Issue:** `find_gpu_bandwidth()` in `core/scoring.py:153` does a linear scan over ~95 GPU entries twice (first for substring match, then for parts match). Called once per model result during search.
- **Files:** `core/scoring.py:153-176`
- **Cause:** Simple dict iteration on every call
- **Improvement path:** Build a trie or normalized lookup table at module load time, or reduce to single pass with combined matching

### 3. SQLite Cache Writes on Every Search
- **Issue:** `cache_db.set_model_cache()` and `set_hardware_snapshot()` each open a new SQLite connection (`_connect()` called inside get/set), write, then close. This is called 15+ times per search result.
- **Files:** `core/cache_db.py:75-86`, `core/cache_db.py:147-158`
- **Cause:** Connection-per-operation pattern instead of connection pooling
- **Improvement path:** Use a module-level connection pool or long-lived connection with reconnection logic

### 4. UI Refresh Causes Full Table Rebuild
- **Issue:** `refresh_table()` in `app.py:2366` clears and rebuilds the entire DataTable from scratch on every filter/sort change, hardware poll, or download status update.
- **Files:** `app.py:2366-2466`
- **Cause:** No incremental row update — full DOM rebuild
- **Improvement path:** Use Textual's `DataTable.update_cell()` for incremental updates where possible

## Fragile Areas

### 1. `app.py` — The Monolith (2466 lines)
- **Files:** `app.py`
- **Why fragile:** Huge single file with implicit dependencies between methods. A change to `refresh_table()` can break `_ensure_download_fields()` which can break `sync_download_jobs_from_service()`.
- **Safe modification:** Isolate changes to individual modal screens first; add tests for any behavior change; extract search/download logic before modifying
- **Test coverage:** 3 test files (~246 lines) for a 2466-line file — ~10% coverage

### 2. Hardware Detection Chain
- **Files:** `core/hardware.py:100-350`
- **Why fragile:** Depends on platform-specific subprocesses (`rocm-smi`, `system_profiler`, `lspci`, `nvidia_smi`) and optional imports (`nvidia_smi`, `pynvml`). Any one of these can fail silently, leaving other detection paths to compensate.
- **Safe modification:** Add logging to each detection path; verify on target platforms; maintain the graceful fallback pattern
- **Test coverage:** `test_hardware_expanded.py` (79 lines) — tests vendor detection functions, not actual hardware interaction

### 3. Download Service Version Compatibility
- **Files:** `downloads/download_service.py:25` (`SERVICE_VERSION = "1.7"`), `downloads/service_client.py:16` (`MIN_SERVICE_VERSION = "1.7"`)
- **Why fragile:** Version mismatch between UI process and background service can silently break downloads. The service client polls `/health` and checks version compatibility, but if the service is outdated, it gets restarted (which kills in-flight downloads).
- **Safe modification:** Add version negotiation on startup; warn user if restart kills active downloads
- **Test coverage:** `test_service_client.py` (13 lines)

### 4. Ollama HTML Scraping
- **Files:** `providers/ollama_provider.py:70-127, 239-260`
- **Why fragile:** Relies on specific HTML table structure, anchor href patterns, and CSS class names on ollama.com
- **Safe modification:** Add integration test with cached HTML samples; monitor for structural changes
- **Test coverage:** `test_ollama_provider_helpers.py` (59 lines)

## Scaling Limits

### 1. In-Memory Search Result Storage
- **Current capacity:** Stores all `all_results` in a single Python list (dict objects for every model)
- **Limit:** With 1000+ models (5 pages of HF results + Ollama results), memory could reach 50-100MB for the results list alone
- **Scaling path:** Implement lazy loading — only hold visible page in memory, fetch on demand

### 2. Download Service Single Worker
- **Current capacity:** One worker thread processes one download at a time (FIFO queue)
- **Limit:** Multiple queued downloads block behind the current one. No concurrency limit for parallel downloads
- **Scaling path:** Add configurable `max_workers` parameter, use `ThreadPoolExecutor` for parallel downloads

### 3. SQLite Concurrency
- **Current capacity:** SQLite connection per operation with thread lock
- **Limit:** Under heavy polling (1.5s interval), download status + hardware metrics + cache DB access could create contention
- **Scaling path:** Use connection pooling with timeout retry

## Dependencies at Risk

### 1. Textual 0.x (>=0.86,<1.0)
- **Risk:** Textual is pre-1.0 software. API breaking changes between minor versions are common. The version pin `>=0.86,<1.0` is already specific, suggesting known compatibility concerns.
- **Impact:** Cannot freely upgrade Textual. New feature adoption requires careful testing.
- **Migration plan:** Pin to working version, monitor Textual changelog before upgrades, run smoke tests after upgrades

### 2. Ollama.com HTML Scraping
- **Risk:** Not an API dependency per se, but the codebase critically depends on Ollama.com's HTML structure remaining stable. No API alternative exists.
- **Impact:** Complete loss of Ollama search if the website redesigns
- **Migration plan:** Add comprehensive error handling, cached HTML test fixtures, and monitoring for search failure rates

## Missing Critical Features

### 1. No HTTP Connection Pooling
- **Problem:** Every provider API call opens a new HTTP connection. HuggingFace's `HfApi` manages its own session, but Ollama, LM Studio, Docker, and MLX providers create new `requests.get()` calls each time.
- **Files:** All provider `search()` methods
- **Blocks:** Efficient multi-query workflows, reduced latency on repeated calls

### 2. No Search Cancellation
- **Problem:** Running a new search while one is in-flight queues behind the current one. The `_search_inflight_signature` check only prevents duplicate searches, not cancellation.
- **Files:** `app.py:1596-1600`
- **Blocks:** Responsive UI during slow searches

### 3. No Offline/Disconnected Mode
- **Problem:** The entire application requires network connectivity. No read-from-cache mode exists for offline operation.
- **Files:** Entire provider layer
- **Blocks:** Usage in air-gapped environments or during network outages

## Test Coverage Gaps

### 1. TUI/Monolith (`app.py`)
- **What's not tested:** 95% of `AIModelViewer` behavior — keyboard actions, search flow, download lifecycle, modal screens, pagination, theme cycling, comparison, compact mode transitions
- **Files:** `app.py` (2466 lines)
- **Risk:** Any change to `app.py` can break core functionality without detection
- **Priority:** High

### 2. Ollama Provider HTML Scraping
- **What's not tested:** The actual HTML parsing logic — only helper functions have tests (`test_ollama_provider_helpers.py`). `search_ollama_models()` and `get_ollama_model_metadata()` have no tests with real or mocked HTML.
- **Files:** `providers/ollama_provider.py:70-127, 195-325`
- **Risk:** Ollama.com HTML changes silently break search
- **Priority:** Medium

### 3. Download Service
- **What's not tested:** `download_service.py` worker loop, cancellation flow, HF API download, subprocess management, version compatibility
- **Files:** `downloads/download_service.py:569-610` (worker loop), `385-483` (HF download), `486-566` (stream download)
- **Risk:** Background download failures go undetected until user reports issues
- **Priority:** Medium

### 4. Error Handling Paths
- **What's not tested:** All the `except Exception: pass` code paths — what happens when hardware detection fails, when HF API is down, when cache DB is corrupted
- **Files:** All error-handling blocks across the codebase
- **Priority:** Medium

---

*Concerns audit: 2026-06-09*
