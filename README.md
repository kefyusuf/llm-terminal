# AI Model Explorer

AI Model Explorer is a Textual-based terminal application for discovering LLM models, checking hardware fit with multi-dimensional scoring, and managing downloads. Supports Ollama, Hugging Face, LM Studio, Docker Model Runner, and MLX.

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

- Python 3.10+
- Internet access for provider searches
- Ollama (optional; required only for local runtime and `ollama pull/run`)
- LM Studio (optional; auto-detected on `localhost:1234`)
- Docker Desktop with Model Runner (optional; auto-detected on `localhost:12434`)

## Installation

```bash
git clone https://github.com/kefyusuf/llm-terminal
cd llm-terminal

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

## Run

### TUI Application

```bash
python main.py
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
python api_server.py              # Start on localhost:8787
python api_server.py --port 9000  # Custom port
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
  app.py                    # Main Textual TUI application
  main.py                   # Entry point
  cli.py                    # CLI commands (system, search, fit, recommend, plan, scores)
  api_server.py             # REST API server (port 8787)
  config.py                 # Pydantic-settings configuration
  hardware.py               # Hardware detection (NVIDIA, AMD, Intel, Apple)
  scoring.py                # 4-dimension scoring engine
  model_intelligence.py     # MoE detection, dynamic quant, size estimation
  themes.py                 # Color theme definitions
  models.py                 # TypedDict schema for model results
  utils.py                  # Shared utilities (fit calc, size est, parsing)
  download_service.py       # Background download HTTP service
  download_manager.py       # Download command builder
  service_client.py         # Download service client
  providers/
    __init__.py             # BaseProvider ABC + provider registry
    ollama_provider.py      # Ollama registry search (HTML scraping)
    hf_provider.py          # Hugging Face API search + download
    lmstudio_provider.py    # LM Studio local server integration
    docker_provider.py      # Docker Model Runner integration
    mlx_provider.py         # Apple Silicon MLX cache integration
  results_presenter.py      # Cell-level Rich markup formatting
  results_view.py           # Result filtering and sorting
  results_layout.py         # Responsive column layout
  results_text.py           # Text truncation and alignment
  search_orchestration.py   # Provider selection, pagination, cache keys
  search_cache.py           # In-memory hardware-aware search cache
  cache_db.py               # SQLite model metadata cache
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

- `terminal_ui/` is kept as a legacy experimental package and is not the primary product path.
- Download service data is stored in `downloads.db`.
- Metadata cache is stored in `cache.db`.
- REST API binds to `127.0.0.1:8787` by default (localhost only).
- Provider auto-detection runs at startup; unavailable providers are silently skipped.
