# External Integrations

**Analysis Date:** 2026-06-09

## APIs & External Services

**Model Search Providers (3rd-party APIs):**
| Service | Usage | Client | Auth |
|---------|-------|--------|------|
| HuggingFace API | Search GGUF models, fetch metadata, download | `huggingface_hub.HfApi` (`providers/hf_provider.py:118`) | `AIMODEL_HF_TOKEN` env var (optional, for rate limits) |
| Ollama Registry | Scrape model library pages for search results | `requests` + BeautifulSoup (`providers/ollama_provider.py:219-238`) | None (anonymous web scraping) |
| Ollama Local API | List installed models (`GET /api/tags`) | `requests` (`providers/ollama_provider.py:37`) | None (localhost only) |

**Local Runtime Detection:**
| Service | Port | Detection Method |
|---------|------|-----------------|
| Ollama | 11434 (default) | psutil process scan + HTTP `/api/tags` |
| LM Studio | 1234 (default) | HTTP `GET /v1/models` (`providers/lmstudio_provider.py:39`) |
| Docker Model Runner | 12434 (default) | HTTP `GET /models` (`providers/docker_provider.py:39`) |
| MLX | Local filesystem | `sysctl hw.optional.arm64` + cache dir scan (`providers/mlx_provider.py:47-54`) |

## Data Storage

**SQLite Databases:**
| Database | Location | Tables | Purpose |
|----------|----------|--------|---------|
| `cache.db` | `data/cache.db` | `model_cache`, `hardware_snapshot` | Caching HF/Ollama metadata, hardware snapshots |
| `downloads.db` | `data/downloads.db` | `jobs` | Download job queue (created by download service) |

**Client:**
- `sqlite3` (stdlib) — Direct SQL access in `core/cache_db.py` and `downloads/download_service.py`

**File Storage:**
- Local filesystem only — Downloaded models stored in `models/` directory (GGUF files)
- MLX models discovered from `~/.cache/huggingface/hub/` and `~/.cache/lm-studio/models/`

**Caching:**
- **Persistent:** SQLite-based (24h TTL for model metadata, 90s TTL for search results)
- **In-memory:** `SearchCache` class in `search/search_cache.py` — 20 entries max, hardware-aware invalidation (checks RAM/VRAM drift)

## Authentication & Identity

**Auth Provider:**
- None built-in. HuggingFace token is optional and used only for higher API rate limits
- Passed via `AIMODEL_HF_TOKEN` -> `HF_TOKEN` env var for download processes
- Download service runs on localhost only (no auth required)

## Background Services

**Download Service:**
- Implementation: `downloads/download_service.py` (796 LOC)
- Protocol: HTTP REST on `127.0.0.1:8765`
- Endpoints:
  - `GET /health` — Health check with version
  - `GET /jobs` — List download jobs
  - `GET /debug/active` — Active download diagnostics
  - `POST /jobs` — Create/upsert download job
  - `POST /jobs/cancel` — Cancel download
  - `POST /jobs/delete` — Delete job record
  - `POST /shutdown` — Graceful shutdown
- Lifecycle: Auto-started by `service_client.ensure_service_running()` on UI mount (`app.py:1195`)
- Architecture: Threaded worker loop polling a SQLite queue

**REST API Server:**
- Implementation: `api_server.py` (339 LOC)
- Protocol: HTTP on `127.0.0.1:8787`
- Endpoints:
  - `GET /health` — Health check
  - `GET /api/v1/system` — Hardware specs
  - `GET /api/v1/models` — Search models (query, provider, limit, sort params)
  - `GET /api/v1/models/top` — Top models
  - `GET /api/v1/models/{name}/plan` — Hardware plan
  - `GET /api/v1/scores/{name}` — Model scores
  - `GET /api/v1/providers` — Available providers

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, no external logging service)

**Logs:**
- **loguru** — Console output (INFO level) + file rotation at `data/logs/app_{time}.log` (DEBUG level, 10MB rotation, 7-day retention)
- `core/logging_.py` — Logger setup, used sparsely across the codebase

## CI/CD & Deployment

**Hosting:**
- Not applicable (terminal application, runs locally)

**CI Pipeline:**
- Not detected (no `.github/workflows/` contents related to CI)

## Environment Configuration

**Required env vars:**
- None strictly required (all have defaults)

**Critical optional env vars:**
| Variable | Default | Purpose |
|----------|---------|---------|
| `AIMODEL_HF_TOKEN` | None | HuggingFace API token for higher rate limits |
| `AIMODEL_HF_SEARCH_LIMIT` | `15` | Results per page |
| `AIMODEL_HF_SEARCH_MAX_PAGES` | `10` | Max pagination pages |
| `AIMODEL_OLLAMA_API_BASE` | `http://localhost:11434` | Ollama local API endpoint |
| `AIMODEL_OLLAMA_TIMEOUT` | `5` | Ollama request timeout (seconds) |
| `AIMODEL_SMOKE` | Not set | Enable smoke-test mode (`"1"` to enable) |

**Secrets location:**
- `.env` file at project root (not committed) — contains `AIMODEL_HF_TOKEN`
- Token forwarded to subprocess via `HF_TOKEN` env var (`downloads/download_service.py:348`)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-06-09*
