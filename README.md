# AI Model Explorer

AI Model Explorer is a terminal-first workspace for discovering, scoring, comparing, and downloading local LLM models across Ollama, Hugging Face, LM Studio, Docker Model Runner, and MLX.

## About

AI Model Explorer helps answer three practical questions:

1. Which models are worth trying?
2. Which models will actually fit this hardware?
3. How can I compare or download them without switching between several tools?

The repository combines a Textual TUI, a scriptable Click CLI, a localhost REST API, hardware-aware scoring, provider orchestration, stale-cache fallback, and a persistent background download service.

## Features

### Model Discovery & Search

- TUI search across **Ollama**, **Hugging Face**, **LM Studio**, **Docker Model Runner**, and **MLX**.
- Filter by provider and use case (coding, chat, vision, reasoning, math, embedding, general).
- Hidden-gem detection for high-download, low-visibility Hugging Face models.
- Parallel provider fan-out with deterministic result grouping.
- Search cancellation so a newer search can supersede in-flight work.
- Capability-driven pagination; continuation is not inferred from result counts.
- TUI stale-cache fallback when live search is unavailable and a matching stale entry exists.
- Provider failures are contained and surfaced instead of silently appearing as ordinary zero-result success.

### Provider Diagnostics

Providers expose both human-readable errors and machine-readable structured diagnostics where supported.

Structured diagnostics include stable metadata such as:

- provider,
- error code,
- retryability,
- optional HTTP status,
- optional retry-after duration.

The TUI/orchestrator preserves this metadata internally. REST exposes it directly, while CLI model-discovery commands surface human-readable warnings on stderr.

Runtime TUI provider detection runs in a background worker, so optional provider probes do not delay search input. The selector starts with Ollama/Hugging Face and updates as detection completes, preserving the active selection.

CPU/GPU discovery also runs in background workers. Cached hardware is displayed immediately; a cold start shows a detection placeholder while the input remains usable. Searches and model details reuse one lazily initialized monitor. Expected detection failures show a retry message instead of closing the application. See [hardware startup behavior](docs/hardware-startup.md).

The opt-in [startup measurement harness](docs/startup-measurements.md) records fresh-process headless startup, input dispatch and first REST provider-list response latency with explicit local budgets and failure accounting.

REST `/api/v1/providers` returns availability immediately with additive `discovery` metadata (`status`, `refreshing`, `stale`, `error`). Availability is provisional while `status` is `pending`, and last-known values are retained during stale/failed refreshes. Hugging Face's initial availability means its remote integration is configured, not that connectivity has been checked. Each API server runs at most one probe at a time and reuses results for 30 seconds. See [provider discovery behavior](docs/provider-discovery.md).

### 4-Dimension Scoring

- **Quality** (0-100): parameter count and quantization quality.
- **Speed** (0-100): estimated tokens/sec based on GPU bandwidth and model size.
- **Fit** (0-100): hardware utilization/fit.
- **Context** (0-100): context-window capacity.
- **Composite score**: use-case-weighted aggregate.

### Hardware Intelligence

- MoE-aware size/VRAM estimation.
- Dynamic quantization planning.
- Token/sec estimation from GPU memory bandwidth.
- NVIDIA, AMD, Apple Silicon, Intel and CPU fallback detection paths.
- NVIDIA detection commits availability/state only after a complete NVML probe, so partial probe failures cannot create a false CUDA result or suppress later vendor fallbacks.
- Multi-GPU-aware hardware snapshots where platform tooling exposes the required data.

### Plan Mode & Comparison

- **Plan Mode** (`P`): reverse hardware analysis for model/quantization choices.
- **Comparison** (`c`/`C`): side-by-side comparison of up to four models.

### Download Management

- Persistent localhost download service.
- Queue, monitor, cancel and delete jobs.
- Bounded parallel workers (`AIMODEL_DOWNLOAD_MAX_WORKERS`, default `2`).
- Ollama pull and Hugging Face download paths.
- Persistent history independent of TUI lifetime.
- Periodic polling failures preserve the last known download state and surface deduplicated stale/recovery status instead of silently appearing current.
- Action-triggered download synchronization also preserves last-known state and reports a failed refresh instead of silently appearing current; unexpected programming failures are not masked.
- Loopback-only transport; optional bearer token for non-health service endpoints.

### REST API

Local HTTP API on `127.0.0.1:8787` by default.

Endpoints:

- `/health`
- `/api/v1/system`
- `/api/v1/models`
- `/api/v1/models/top`
- `/api/v1/models/{name}/plan`
- `/api/v1/scores/{name}`
- `/api/v1/providers`

`/api/v1/models` currently searches **Ollama and Hugging Face**. Successful responses include additive `errors` and `structured_errors` arrays so callers can distinguish provider failure from a genuine zero-result search while still receiving partial results.

### CLI

Rich terminal commands for system information, search, fit analysis, recommendations, planning and scoring.

CLI search/fit/recommend provider failures are printed to **stderr**. `recommend --json` keeps stdout as valid JSON for scripting.

Run `ai-model-explorer-cli doctor --json` for shareable storage, runtime and TLS diagnostics, or add `--offline` to skip network probes. The default total network wait budget is three seconds. See [doctor diagnostics](docs/doctor.md) for exit codes, privacy and troubleshooting.

Model quality, throughput, fit and context scores are heuristic estimates. CLI `scores`, TUI comparison and additive REST/recommendation JSON provenance explain their inputs and defaults. The context score is a model-size proxy, not supported token capacity. See [score estimates](docs/score-estimates.md).

Search and hardware planning expose additive schema-1 JSON; saved-search
comparison preserves recorded identity, scores and errors without another
network search. See [model exports and exact artifact handoff](docs/model-exports.md).
The existing `recommend --json` array remains compatible.

Saved runtime declarations can drive explicit context/concurrency/placement
scenarios through `plan --facts`; these retain unknowns and estimate assumptions.
See [memory scenarios](docs/metadata-memory-scenarios.md). Developers can collect
bounded repeated measurements from an already prepared dedicated local Ollama
with the [calibration tool](docs/runtime-calibration.md); no runtime or model is
downloaded by that tool, and no genuine calibration corpus is bundled.

Hugging Face downloads preserve the selected file's resolved commit when available,
use server-owned disk/path plans and validate declared byte/digest identity on
completion. Legacy unknowns remain explicit. See
[download identity and plans](docs/hf-download-identity.md) for the acquisition contract.

### Theming

- default
- dracula
- nord
- solarized
- monokai

Cycle themes at runtime with `t`.

## Requirements

- Python 3.10-3.14.
- CI verifies Python 3.12 and 3.14 on Ubuntu and Windows, plus a macOS/Python 3.12 bootstrap, verify and smoke lane.
- Internet access for live remote provider searches.
- Ollama optional; required for Ollama local runtime/pull operations.
- LM Studio optional; auto-detected on `localhost:1234` by default.
- Docker Desktop Model Runner optional; auto-detected on `localhost:12434` by default.
- macOS Apple Silicon for MLX runtime/cache detection.

The TUI can reuse a matching stale search-cache entry when a live search is unavailable, but this is not a general offline mirror of every provider.

## Installation

```bash
git clone https://github.com/kefyusuf/llm-terminal
cd llm-terminal
python scripts/dev.py bootstrap
python scripts/dev.py verify
python scripts/dev.py coverage
python scripts/dev.py smoke
```

Use a supported Python 3.10-3.14 interpreter for bootstrap. On Windows that can be `py -3.14` or another selected interpreter. After bootstrap, prefer the project virtualenv over a random global Python on `PATH`.

`scripts/dev.py bootstrap` creates or reuses `.venv`. Windows and Linux install from committed platform-specific development lock files. macOS currently installs through `requirements/requirements-dev.txt`, the compatibility wrapper that resolves the human-edited dependency intent from `requirements-dev.in`, until native macOS lock files are committed. `scripts/dev.py lock` remains intentionally Windows/Linux-only rather than treating another platform's lock as a macOS lock. Edit `requirements/requirements.in` and `requirements/requirements-dev.in` for dependency intent; bootstrap does not regenerate locks.

`scripts/dev.py verify` runs the required local checks: pytest, import smoke and Ruff.

`scripts/dev.py coverage` runs the deterministic pytest-cov lane and enforces the threshold configured in `pyproject.toml`. The canonical Ubuntu/Python 3.12 lane now measures **67.44% total coverage** (`5095` statements, `1659` missed) and enforces a **60%** merge floor. Focused deterministic tests raised `downloads/runner.py` from **24% to 90%**, `downloads/service_client.py` from **46% to 90%**, and `core/hardware.py` from **44% to 52%** while pinning concrete correctness contracts. The staged 50% → 55% → 60% coverage ratchet is complete; future increases should remain evidence-driven and must not rely on production-code exclusions.

`scripts/dev.py smoke` runs bounded/offline-safe smoke checks for the CLI, REST API, TUI startup path and download service.

Package CI verifies wheel installations on Linux/Windows/macOS with Python 3.10–3.14 and sdist installations on each OS with Python 3.12, using fresh environments outside the checkout. It checks installed module origins, console scripts, REST and download-service startup and records resolved dependencies. See [installed package validation](docs/installed-package-validation.md) for the checks, local evidence and limits.

The [operation-specific support matrix](docs/support-matrix.md) separates tested
installation/startup from live download recovery and unverified inference.
[HF download plans](docs/hf-download-identity.md) preview exact identity,
destination and known disk budget before queueing; [managed file removal](docs/managed-download-removal.md)
preserves unrelated models and SDK caches. Current CI [builds one candidate](docs/build-once-candidates.md)
for all installation consumers. [Release readiness](docs/release-readiness.md)
records the remaining runtime, pilot, downgrade and publication gates; a green
package build is not a stable-release or model-inference certification.

The current verify baseline contains **600+ tests**.

## Run

### TUI

```bash
.venv/Scripts/python.exe main.py  # Windows
# or
.venv/bin/python main.py          # Linux/macOS
```

Installed entry point:

```bash
ai-model-explorer
```

### CLI

```bash
ai-model-explorer-cli system
ai-model-explorer-cli search "llama" -n 5
ai-model-explorer-cli fit --perfect -n 5
ai-model-explorer-cli recommend -u coding
ai-model-explorer-cli recommend -u coding --json
ai-model-explorer-cli plan "llama-3-8b"
ai-model-explorer-cli scores "llama-70b"
```

### REST API

```bash
python -m api_server
python -m api_server --port 9000
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AIMODEL_HF_TOKEN` | - | Canonical Hugging Face read-only token; standard `HF_TOKEN` is accepted as fallback |
| `AIMODEL_HF_MODELS_DIR` | OS-specific user-data `models/` | Hugging Face model download directory |
| `AIMODEL_DOWNLOAD_SERVICE_HOST` | `127.0.0.1` | Download-service bind/client host; non-loopback hosts are rejected |
| `AIMODEL_DOWNLOAD_SERVICE_PORT` | `8765` | Download-service port |
| `AIMODEL_DOWNLOAD_SERVICE_TOKEN` | - | Optional bearer token for download-service endpoints except `/health` |
| `AIMODEL_DOWNLOAD_MAX_WORKERS` | `2` | Parallel download worker count |
| `AIMODEL_HF_SEARCH_LIMIT` | `15` | Hugging Face results per page |
| `AIMODEL_HF_SEARCH_MAX_PAGES` | `10` | Maximum Hugging Face pages |
| `AIMODEL_OLLAMA_API_BASE` | `http://localhost:11434` | Ollama local API base |
| `AIMODEL_UI_MODE` | `compact` | `compact` or `comfortable` |
| `AIMODEL_THEME` | `default` | Color theme |
| `LMSTUDIO_HOST` | `http://localhost:1234` | LM Studio API address |
| `DOCKER_MODEL_RUNNER_HOST` | `http://localhost:12434` | Docker Model Runner API address |

### Example `.env`

```env
AIMODEL_HF_TOKEN=hf_your_token_here
# HF_TOKEN=hf_your_token_here
# AIMODEL_HF_MODELS_DIR=/custom/path/models
# AIMODEL_DOWNLOAD_SERVICE_TOKEN=replace-with-a-long-random-secret
AIMODEL_UI_MODE=compact
AIMODEL_THEME=nord
```

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `/` | Focus search |
| `r` | Refresh current search |
| `p` | Cycle provider filter |
| `[` | Previous page (Hugging Face) |
| `]` | Next page (Hugging Face) |
| `u` | Cycle use-case filter |
| `s` | Cycle sort mode |
| `f` | Cycle fit filter |
| `P` | Open plan mode |
| `c` | Toggle selected model in comparison set |
| `C` | Show comparison modal (2+ models) |
| `v` | Toggle compact/comfortable view |
| `h` | Toggle hidden-gems filter |
| `t` | Cycle theme |
| `q` | Quit |

## Project Structure

```text
llm-terminal/
  main.py                   # Primary TUI entry point
  app/viewer.py             # Runtime AIModelViewer/provider selector
  tui_app.py                # Base Textual application
  cli.py                    # Click CLI
  api_server.py             # Local REST API
  config.py                 # Pydantic settings
  app/                      # Modals, widgets, TUI state/download support
  core/                     # Hardware, scoring, cache, HTTP, errors, intelligence
  providers/                # Provider implementations + capabilities
  search/                   # Search cache/orchestration/cancellation/pagination
  results/                  # Result layout/presentation/filtering
  downloads/                # Download service, API, store, runner, client/helpers
  requirements/             # Dependency intent + committed Windows/Linux locks
  scripts/                  # Dev/release/maintenance tools
  terminal_ui/              # Theme/style assets and isolated legacy internals
  tests/                    # 600+ deterministic tests + optional live tests
  docs/                     # Maintained project/process documentation
  .planning/                # Active roadmap and current codebase baseline
```

## Scoring System

| Use Case | Quality | Speed | Fit | Context |
|---|---:|---:|---:|---:|
| Chat | 0.25 | 0.35 | 0.25 | 0.15 |
| Coding | 0.35 | 0.30 | 0.20 | 0.15 |
| Reasoning | 0.55 | 0.15 | 0.15 | 0.15 |
| Vision | 0.30 | 0.25 | 0.25 | 0.20 |
| Math | 0.45 | 0.20 | 0.20 | 0.15 |
| Embedding | 0.30 | 0.40 | 0.20 | 0.10 |
| General | 0.30 | 0.25 | 0.25 | 0.20 |

Speed approximation:

```text
tok/s = (GPU bandwidth GB/s / model size GB) × efficiency factor
```

Inference mode influences the efficiency factor (GPU, GPU+CPU offload, CPU-only).

## Development Roadmap

The active post-hardening roadmap is in `.planning/roadmap.md`.

Current priorities:

1. Transparent recommendation estimates, bounded startup/provider detection and clean package installation outside the checkout,
2. exact model-artifact provenance and real download/cancellation/recovery acceptance,
3. an evidence-based platform/provider support matrix,
4. a tested package release, pilot and upgrade/recovery process,
5. metadata-aware calibration, scriptable output and measured TUI performance improvements.

The [industry comparison and release assessment](docs/industry-readiness.md) explains the rationale and proposed release gates. These are planned milestones, not claims that public-release readiness has already been achieved.

Documentation maintenance rules are in `docs/maintenance.md`.

## Notes

- `terminal_ui/` is not the primary application entry point; it mainly contains theme/style assets plus isolated legacy internals.
- Runtime databases, logs and Hugging Face model files use OS-specific per-user application-data locations by default; configuration can override them.
- The download service remains loopback-only. LAN/public transport is intentionally rejected until an authenticated TLS design exists.
- REST API binds to `127.0.0.1:8787` by default and is intended for local use.
- Optional provider detection failures are contained so unavailable local runtimes do not prevent other providers from working.
- Ollama remote discovery still depends on `ollama.com` HTML structure; parser resilience is an active roadmap priority.
