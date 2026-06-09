# Codebase Structure

**Analysis Date:** 2026-06-09

## Directory Layout

```
ai-model-explorer/
├── app.py              # Main TUI application (2466 lines — monolith)
├── main.py             # Entry point — runs AIModelViewer
├── cli.py              # Click CLI with 8 commands (398 lines)
├── api_server.py       # REST API on port 8787 (339 lines)
├── config.py           # Pydantic Settings with env overrides (62 lines)
├── pyproject.toml      # Build, lint, test, coverage config
│
├── core/               # Domain logic — hardware, scoring, caching (1627 lines)
│   ├── hardware.py     #   HardwareMonitor + GPU detection (389 lines)
│   ├── scoring.py      #   4-dimension scoring engine (475 lines)
│   ├── model_intelligence.py  # MoE detection, quant selection (334 lines)
│   ├── models.py       #   ModelResult TypedDict (54 lines)
│   ├── utils.py        #   Name parsing, fit calculation (183 lines)
│   ├── cache_db.py     #   SQLite metadata cache (158 lines)
│   └── logging_.py     #   loguru setup (33 lines)
│
├── providers/          # Search backends (1162 lines)
│   ├── __init__.py     #   BaseProvider ABC + registry (154 lines)
│   ├── hf_provider.py  #   HuggingFace Hub search (272 lines)
│   ├── ollama_provider.py     # Ollama registry scraping (325 lines)
│   ├── lmstudio_provider.py   # LM Studio local API (123 lines)
│   ├── docker_provider.py     # Docker Model Runner (139 lines)
│   └── mlx_provider.py        # Apple Silicon MLX (149 lines)
│
├── downloads/          # Download management (1376 lines)
│   ├── download_service.py    # Background HTTP service (796 lines)
│   ├── service_client.py      # HTTP client for service (197 lines)
│   ├── download_manager.py    # Download command builder (41 lines)
│   ├── download_history.py    # History helpers (58 lines)
│   ├── download_lifecycle.py  # Lifecycle management (134 lines)
│   ├── download_status.py     # Status/state helpers (92 lines)
│   └── hf_downloader.py       # HF-specific download logic (58 lines)
│
├── search/             # Search orchestration + caching (212 lines)
│   ├── search_orchestration.py  # Provider-aware search flow (123 lines)
│   └── search_cache.py          # In-memory cache with HW invalidation (89 lines)
│
├── results/            # Results table presentation (589 lines)
│   ├── results_layout.py       # Responsive column widths (167 lines)
│   ├── results_presenter.py    # Color-coded cell markup (241 lines)
│   ├── results_text.py         # Truncation, alignment (62 lines)
│   └── results_view.py         # Filter + sort engine (119 lines)
│
├── terminal_ui/        # Legacy Obsidian Console (769 lines)
│   ├── app.py          #   ObsidianConsole — experimental workspace (631 lines)
│   └── themes.py       #   Color theme definitions (137 lines)
│
├── scripts/            # Development/CI scripts
│   ├── dev.py
│   ├── release_check.py
│   ├── release_check_helpers.py
│   ├── reset_downloads.py
│   └── reset_all.bat
│
├── tests/              # 34 test files, 3172 lines
│   ├── conftest.py     #   Pytest config + --run-live flag
│   ├── test_scoring.py #   303 lines — comprehensive scoring tests
│   ├── test_model_intelligence.py  # 214 lines
│   ├── test_providers_extended.py  # 180 lines
│   ├── test_api_server.py          # 168 lines
│   ├── test_download_service_store.py  # 154 lines
│   ├── test_download_lifecycle.py  # 118 lines
│   ├── test_scoring_ui.py          # 205 lines
│   ├── test_dev_script.py          # 424 lines
│   └── ... (26 more files)
│
├── data/               # Runtime data (git-ignored?)
│   ├── cache.db        #   Model metadata + hardware snapshot cache
│   └── downloads.db    #   Download job queue
│
├── models/             # Downloaded GGUF model files
├── docs/               # Documentation
│   ├── structure.md
│   ├── archive/
│   └── superpowers/
│
├── requirements/       # Pinned dependency files
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── requirements-linux.txt
│   ├── requirements-windows.txt
│   └── *.in (source files)
│
└── .env                # Environment variables (not committed)
```

## Directory Purposes

**`core/`** — Domain Business Logic:
- Purpose: Hardware detection, model scoring, MoE intelligence, caching, utility functions
- Contains: Domain services with no dependencies on UI or provider-specific code
- Key files: `hardware.py` (hardware detection for all vendors), `scoring.py` (scoring engine + GPU bandwidth DB)

**`providers/`** — Search Backends:
- Purpose: Pluggable model search implementations following `BaseProvider` ABC
- Contains: One file per provider, each with `detect()`, `search()`, `list_installed()`
- Key files: `__init__.py` (registry, lazy imports), `hf_provider.py` (primary HF search)

**`downloads/`** — Download Management:
- Purpose: Background download orchestration via subprocess HTTP service
- Contains: HTTP server + client, state machine, SQLite job queue
- Key files: `download_service.py` (796-line HTTP+worker server), `service_client.py` (UI-side client)

**`search/`** — Search Pipeline:
- Purpose: Orchestrate multi-provider search, cache results
- Contains: Cache-aware search dispatch, pagination helpers

**`results/`** — Results Presentation:
- Purpose: Filter, sort, layout, and colorize model results for DataTable rendering
- Contains: Pure formatting functions with no UI dependencies

**`tests/`** — Test Suite:
- Purpose: 34 test files, 3172 lines covering scoring, hardware, providers, downloads, API, UI
- Contains: Unit tests + smoke tests; live integration tests behind `--run-live` flag
- Key files: `conftest.py` (sys.path setup, --run-live flag, marker registration)

## Key File Locations

**Entry Points:**
- `main.py` — Primary TUI entry point (`ai-model-explorer`)
- `cli.py` — CLI entry point (`ai-model-explorer-cli`)
- `api_server.py` — REST API (`python -m api_server`)
- `downloads/download_service.py` — Background service (auto-launched)

**Configuration:**
- `pyproject.toml` — Project metadata, dependencies, ruff, mypy, pytest, coverage
- `config.py` — Runtime settings via Pydantic
- `.env` — Environment variable overrides (not committed)
- `requirements/*.txt` — Pinned dependency versions

**Core Logic:**
- `app.py` — Main TUI application (2466 lines)
- `core/scoring.py` — Scoring engine (475 lines)
- `core/hardware.py` — Hardware detection (389 lines)
- `core/model_intelligence.py` — MoE/quant intelligence (334 lines)
- `providers/hf_provider.py` — HuggingFace search (272 lines)
- `providers/ollama_provider.py` — Ollama search (325 lines)

**Testing:**
- `tests/` directory, 34 files, 3172 lines total
- Configuration in `[tool.pytest.ini_options]` in `pyproject.toml`

## Naming Conventions

**Files:**
- `snake_case.py` — All Python files
- Module `__init__.py` files present in all packages

**Functions:**
- `snake_case` — All functions and methods
- Private functions prefixed with `_` (e.g., `_select_preferred_gguf`, `_cpu_count`)

**Variables:**
- `snake_case` — All variables
- Module-level constants: `UPPER_CASE` (e.g., `SERVICE_VERSION`, `GPU_BANDWIDTH`)

**Types:**
- `PascalCase` — Classes (`AIModelViewer`, `HardwareMonitor`, `BaseProvider`)
- `PascalCase` — TypedDicts (`ModelResult`), dataclasses (`Scores`, `Theme`)
- `_T` suffix — TypeAlias conventions (e.g., `_TruncatePlain`, `_AlignPlain`)

## Where to Add New Code

**New Feature (e.g., new provider):**
- Add provider class in `providers/{name}_provider.py` extending `BaseProvider`
- Register in `providers/__init__.py:get_all_provider_classes()` and `get_provider_filter_labels()`
- Add UI filter label in `providers/__init__.py:_FILTER_TO_SLUGS` (if not in `search/search_orchestration.py`)

**New Scoring Dimension:**
- Add scoring function in `core/scoring.py`
- Add weight in `USE_CASE_WEIGHTS` dict
- Add field to `Scores` dataclass
- Update `ModelResult` TypedDict in `core/models.py`
- Add column to `results_presenter.py` markup functions
- Add to comparison modal in `app.py:627-655`

**New UI Modal:**
- Create new `ModalScreen` subclass in `app.py` following `ModelDetailModal` pattern
- Add action method to `AIModelViewer` (e.g., `action_open_plan_mode`)
- Add keybinding in `BINDINGS` list

**New CLI Command:**
- Add Click command in `cli.py` following `@cli.command()` pattern
- Wire into `pyproject.toml` if needed

**New API Endpoint:**
- Add handler method to `ModelAPIHandler` in `api_server.py`
- Register route in `do_GET()` dispatch

## Special Directories

**`data/`:** Runtime SQLite databases created on first run. Contains `cache.db` and `downloads.db`. Should be added to `.gitignore`.

**`models/`:** Downloaded GGUF model files. Large binary files — should be excluded from version control.

**`models/.cache/`:** HuggingFace Hub cache for downloaded model metadata.

**`terminal_ui/`:** Contains a legacy, unused application (`ObsidianConsole` in `app.py:329`). Not connected to the main entry point. Kept as historical reference.

---

*Structure analysis: 2026-06-09*
