# Technology Stack

**Analysis Date:** 2026-06-09

## Languages

**Primary:**
- Python >=3.10 — All application code, from TUI to download service to API server

**Secondary:**
- CSS (Textual TCSS) — `terminal_ui/theme.tcss` for styling the legacy Obsidian Console widget

## Runtime

**Environment:**
- CPython 3.10+ (required by `pyproject.toml`)

**Package Manager:**
- pip (via `requirements.in` / `requirements.txt`)
- pip-compile (via `requirements/*.in` -> `requirements/*.txt`)
- Lockfile: `requirements.txt`, `requirements-dev.txt`, plus platform-specific variants in `requirements/`
- Virtual envs detected: `.venv/`, `.venv314/`, `.venv.py313-broken/`

## Frameworks

**Core:**
- **Textual** `>=0.86,<1.0` — Terminal UI framework powering the main `AIModelViewer` app in `app.py` (2466 LOC)
- **Rich** `>=13.9,<14.0` — Terminal formatting, used in `cli.py` for tables and markup
- **Click** `>=8.0,<9.0` — CLI framework in `cli.py` (8 commands: info, system, search, fit, recommend, plan, scores, cache-clear, cache-stats, version)
- **Pydantic Settings** `>=2.0,<3.0` — `config.py` `Settings` class with `.env` support via `AIMODEL_` prefix

**Web/HTTP:**
- **requests** `>=2.32,<3.0` — HTTP client for HuggingFace API, Ollama registry scraping, LM Studio/Docker APIs
- **urllib** (stdlib) — HTTP client for download service (`service_client.py`) and API server
- **http.server** (stdlib) — HTTP server for download service (`downloads/download_service.py`) and REST API (`api_server.py`)

**Scraping:**
- **BeautifulSoup4** `>=4.12,<5.0` — HTML parsing for `ollama.com` library page scraping

**Data & ML Integration:**
- **huggingface_hub** `>=0.27,<1.0` — HuggingFace API for model search and download
- **nvidia-ml-py** `>=12.560,<13.0` — NVIDIA GPU detection (VRAM, GPU name)

**System/Hardware:**
- **psutil** `>=5.9,<7.0` — CPU, RAM, process monitoring with graceful fallback

**Other:**
- **loguru** `>=0.7,<1.0` — Structured logging (`core/logging_.py`)
- **pyperclip** `>=1.8,<2.0` — Clipboard integration for copy-command feature

**Testing:**
- **pytest** (via `[tool.pytest.ini_options]`) — 34 test files, 3172 lines
- **coverage** — `fail_under = 40`

**Linting/Formatting:**
- **Ruff** — Linter + formatter, target `py310`, line-length `100`
  - Rules: E, W, F, I, UP, B, C4, SIM, RUF
  - Isort: known-first-party for app modules
- **Mypy** — Static type checking, `strict = false`, `ignore_missing_imports = true`

## Key Dependencies

**Critical:**
| Package | Version | Why It Matters |
|---------|---------|----------------|
| `textual` | >=0.86,<1.0 | Entire TUI depends on it. 0.x maturity — API churn risk |
| `huggingface_hub` | >=0.27,<1.0 | Primary model source. Search + download both depend on it |
| `requests` | >=2.32,<3.0 | Ubiquitous HTTP client across all providers and web scraping |

**Infrastructure:**
| Package | Version | Purpose |
|---------|---------|---------|
| `psutil` | >=5.9,<7.0 | Hardware monitoring with graceful `ImportError` fallback |
| `nvidia-ml-py` | >=12.560 | NVIDIA VRAM / GPU detection |
| `beautifulsoup4` | >=4.12 | Ollama.com HTML scraping |
| `pydantic-settings` | >=2.0 | Type-safe configuration |
| `loguru` | >=0.7 | Structured logging with rotation |
| `pyperclip` | >=1.8 | Model command copy to clipboard |
| `click` | >=8.0 | CLI framework |

## Configuration

**Environment:**
- `.env` file with `AIMODEL_` prefix (managed by Pydantic Settings in `config.py`)
- `AIMODEL_HF_TOKEN` — HuggingFace API token (optional, increases rate limits)
- `AIMODEL_HF_SEARCH_LIMIT` — Results per page (default 15)
- `AIMODEL_HF_SEARCH_MAX_PAGES` — Max pages to fetch (default 10)
- `AIMODEL_OLLAMA_API_BASE` — Ollama server URL (default `http://localhost:11434`)
- `AIMODEL_OLLAMA_TIMEOUT` — Ollama request timeout (default 5s)
- `AIMODEL_SEARCH_CACHE_TTL_SECONDS` — Cache TTL (default 90s)
- `AIMODEL_SEARCH_CACHE_MAX_ENTRIES` — Max cache entries (default 20)

**Build:**
- `pyproject.toml` — Build config, ruff, mypy, pytest, coverage all configured here
- `requirements/requirements.in` — Base deps
- `requirements/requirements-dev.in` — Dev deps
- Platform-specific `requirements-linux.txt` and `requirements-windows.txt`

## Platform Requirements

**Development:**
- Python >= 3.10
- pip + virtualenv
- A HuggingFace token (optional, but recommended for rate limits)
- Ollama (optional, for local model runtime features)
- NVIDIA GPU with CUDA (optional, for accelerated inference)

**Production:**
- Run as a terminal application (TUI) or CLI
- Background download service process (launched automatically)
- Can be accessed programmatically via REST API on port 8787

---

*Stack analysis: 2026-06-09*
