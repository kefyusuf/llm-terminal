# AI Model Explorer

AI Model Explorer is a Textual terminal app for discovering, scoring, comparing, and downloading local LLM models across Ollama, Hugging Face, LM Studio, Docker Model Runner, and MLX.

## About

AI Model Explorer is a terminal-first workspace for people who run local LLMs and want faster answers to three practical questions: which models are worth trying, which ones will actually fit their hardware, and how to download or compare them without bouncing between multiple tools.

The repo combines model discovery, hardware-aware scoring, plan-mode analysis, local provider integration, and download orchestration in one place. It is designed for developers and power users who want a Textual UI, a scriptable CLI, and a local REST API over the same core model intelligence.

## Features

### Model Discovery & Search
- Search across 5 providers: **Ollama**, **Hugging Face**, **LM Studio**, **Docker Model Runner**, **MLX**
- Filter by provider, use case (coding, chat, vision, reasoning, math, embedding, general)
- Hidden-gem detection for high-download, low-visibility HF models
- Pagination support for Hugging Face results

### 4-Dimension Scoring System
- **Quality** (0-100): Parameter count and quantization quality
- **Speed** (0-100): Estimated tokens/sec based on GPU bandwidth and model size
- **Fit** (0-100): VRAM utilization efficiency (sweet spot: 50-80%)
- **Context** (0-100): Context window capacity
- **Composite score**: Use-case-weighted average (e.g., chat prioritizes speed, reasoning prioritizes quality)

### Hardware Intelligence
- **MoE awareness**: Auto-detects Mixture-of-Experts models (Mixtral, DeepSeek-V2/V3, etc.) and adjusts VRAM calculations
- **Dynamic quantization**: Automatically selects the best quantization level (Q8_0 to Q2_K) for your hardware
- **Speed estimation**: Token/sec prediction based on GPU memory bandwidth (80+ GPUs in lookup table)
- **Multi-GPU support**: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), Intel Arc (SYCL)

### Plan Mode & Comparison
- **Plan Mode** (`P`): Reverse hardware analysis — see what GPU you need for any model at any quantization level
- **Comparison** (`c`/`C`): Side-by-side comparison of up to 4 models with full scoring breakdown

### Download Management
- Queue, monitor, cancel, and delete downloads through a background HTTP service
- Track download history with status and details
- Support for Ollama pull and Hugging Face GGUF downloads

### Theming
- 5 built-in themes: default, dracula, nord, solarized, monokai
- Runtime toggle with `t` key

### REST API
- Local HTTP API on port 8787 for programmatic access
- Endpoints: `/health`, `/api/v1/system`, `/api/v1/models`, `/api/v1/models/top`, `/api/v1/models/{name}/plan`, `/api/v1/scores/{name}`, `/api/v1/providers`

### CLI
- Rich terminal output for search, fit analysis, recommendations, and scoring
- JSON output support for scripting

## Requirements

- Python 3.10-3.14
- Tested in CI: Python 3.12 and Python 3.14
- Internet access for provider searches
- Ollama (optional; required only for local runtime and `ollama pull/run`)
- LM Studio (optional; auto-detected on `localhost:1234`)
- Docker Desktop with Model Runner (optional; auto-detected on `localhost:12434`)

## Installation

```bash
git clone https://github.com/kefyusuf/llm-terminal
cd llm-terminal
python scripts/dev.py bootstrap
python scripts/dev.py verify
python scripts/dev.py smoke
```

Use a supported Python 3.10-3.14 interpreter for the bootstrap command. On Windows,
that can be `py -3.14` or another selected interpreter; after bootstrap, prefer the
project virtualenv instead of a random global `python` on your `PATH`.

`scripts/dev.py bootstrap` creates or reuses `.venv` and installs from the committed
platform-specific development lock file. Edit `requirements/requirements.in` and
`requirements/requirements-dev.in` for dependency intent; bootstrap does not resolve or
regenerate locks. Canonical lock maintenance stays on Python 3.12 even though the runtime
support window is 3.10-3.14.

`scripts/dev.py verify` runs the required local checks in order: `pytest -q`, import smoke,
and `ruff check .`.

`scripts/dev.py smoke` runs the offline-safe process smoke checks for the CLI, API server,
TUI startup path, and download service.

## Run

### TUI Application

```bash
.venv/Scripts/python.exe main.py  # Windows
# or
.venv/bin/python main.py          # Linux/macOS
```

Or with the installed script entrypoint:

```bash
ai-model-explorer
```

### CLI Commands

```bash
ai-model-explorer-cli system              # Show hardware info
ai-model-explorer-cli search "llama" -n 5 # Search models
ai-model-explorer-cli fit --perfect -n 5  # Find best-fitting models
ai-model-explorer-cli recommend -u coding # Get coding model recommendations
ai-model-explorer-cli plan "llama-3-8b"   # Hardware requirements analysis
ai-model-explorer-cli scores "llama-70b"  # Detailed scoring breakdown
```

### REST API Server

```bash
python -m api_server              # Start on localhost:8787
python -m api_server --port 9000  # Custom port
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AIMODEL_HF_TOKEN` | - | Hugging Face read-only token for higher rate limits |
| `AIMODEL_HF_SEARCH_LIMIT` | 15 | HF results per page |
| `AIMODEL_HF_SEARCH_MAX_PAGES` | 10 | Max HF pages |
| `AIMODEL_OLLAMA_API_BASE` | `http://localhost:11434` | Ollama API base URL |
| `AIMODEL_UI_MODE` | `compact` | UI density: `compact` or `comfortable` |
| `AIMODEL_THEME` | `default` | Color theme: `default`, `dracula`, `nord`, `solarized`, `monokai` |
| `LMSTUDIO_HOST` | `http://localhost:1234` | LM Studio API address |
| `DOCKER_MODEL_RUNNER_HOST` | `http://localhost:12434` | Docker Model Runner API address |

### Example `.env`

```env
AIMODEL_HF_TOKEN=hf_your_token_here
AIMODEL_UI_MODE=compact
AIMODEL_THEME=nord
```

## Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `/` | Focus search |
| `r` | Refresh current search |
| `p` | Cycle provider filter |
| `[` | Previous page (HF) |
| `]` | Next page (HF) |
| `u` | Cycle use-case filter |
| `s` | Cycle sort mode (composite, speed, quality, name, downloads) |
| `f` | Cycle fit filter |
| `P` | Open plan mode (hardware analysis for selected model) |
| `c` | Toggle selected model into comparison set |
| `C` | Show comparison modal (need 2+ models) |
| `v` | Toggle compact/comfortable view |
| `h` | Toggle hidden gems filter |
| `t` | Cycle color theme |
| `q` | Quit |

## Project Structure

```text
llm-terminal/
  tui_app.py                # Main Textual TUI application
  main.py                   # Entry point
  cli.py                    # CLI commands (system, search, fit, recommend, plan, scores)
  api_server.py             # REST API server (port 8787)
  config.py                 # Pydantic-settings configuration
  core/                     # Shared cache, hardware, logging, model metadata, scoring, and helpers
  requirements/            # Runtime/dev intent + committed platform locks
  downloads/               # Download state, lifecycle, command builder, service client, HF downloader
  results/                 # Table layout, formatting, filtering helpers
  search/                  # Search cache and provider orchestration
  scripts/                 # Dev, release, and maintenance utilities (Python + batch)
  terminal_ui/             # Theme/style assets plus isolated legacy UI internals
  providers/
    __init__.py             # BaseProvider ABC + provider registry
    ollama_provider.py      # Ollama registry search (HTML scraping)
    hf_provider.py          # Hugging Face API search + download
    lmstudio_provider.py    # LM Studio local server integration
    docker_provider.py      # Docker Model Runner integration
    mlx_provider.py         # Apple Silicon MLX cache integration
  tests/                    # Test suite (261 tests)
```

## Scoring System

The scoring engine evaluates models on 4 dimensions, then computes a use-case-weighted composite:

| Use Case | Quality | Speed | Fit | Context |
|---|---|---|---|---|
| Chat | 0.25 | 0.35 | 0.25 | 0.15 |
| Coding | 0.35 | 0.30 | 0.20 | 0.15 |
| Reasoning | 0.55 | 0.15 | 0.15 | 0.15 |
| Vision | 0.30 | 0.25 | 0.25 | 0.20 |
| Math | 0.45 | 0.20 | 0.20 | 0.15 |
| Embedding | 0.30 | 0.40 | 0.20 | 0.10 |
| General | 0.30 | 0.25 | 0.25 | 0.20 |

**Speed formula**: `tok/s = (GPU_bandwidth_GB/s / model_size_GB) × efficiency_factor`

Where efficiency depends on inference mode: GPU (0.55), GPU+CPU offload (0.30), CPU-only (0.18).

## Notes

- `terminal_ui/` now exists mainly for theme/style assets; the legacy experimental UI remains isolated there.
- Download service data is stored in `data/downloads.db`.
- Metadata cache is stored in `data/cache.db`.
- REST API binds to `127.0.0.1:8787` by default (localhost only).
- Provider auto-detection runs at startup; unavailable providers are silently skipped.
