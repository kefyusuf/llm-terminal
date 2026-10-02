# Runtime Validation and Continuation Baseline

Validation date: 2026-10-02. Source baseline: `32fe324` on local `main`.

## Environment and Checks

- Windows, Python 3.12.14; `.venv` created by `scripts/dev.py bootstrap` from the committed Windows development lock and an editable project installation.
- `scripts/dev.py verify`: passed all 682 collected tests, import smoke, and Ruff.
- `scripts/dev.py smoke`: passed CLI system inspection, REST transport health, headless Textual startup, and download-service health/jobs checks.
- CI's staged mypy command: passed for 32 source files. This is a scoped check, not whole-project strict typing.
- Actual hardware probe identified AMD Ryzen 5 7500F, approximately 31.4 GB RAM, and NVIDIA GeForce RTX 4060 Ti with 16 GB VRAM/CUDA. Free memory is transient.
- Normal REST server returned HTTP 200 for `/health` and `/api/v1/system` using the real hardware monitor.
- `/api/v1/providers` also returned HTTP 200 when allowed a 45-second client deadline; it reported only Hugging Face as available.
- A separate download-service process returned health version `1.8` and an empty jobs list from a fresh SQLite database.
- The normal runtime TUI mounted successfully in Textual's headless driver, connected to the real download service, completed a remote search attempt, and switched to comfortable mode with `v`.
- Initial Hugging Face HTTPS access failed with `CERTIFICATE_VERIFY_FAILED` (self-signed certificate in the chain). Using a temporary PEM bundle exported from Windows' existing trusted Root/CA stores through `REQUESTS_CA_BUNDLE`, CLI live search returned two models. TLS verification remained enabled.
- With the same trusted-CA bundle, the normal TUI searched `Qwen2.5-0.5B-Instruct`, received two real Hugging Face results, rendered two table rows with no search error, and switched to comfortable mode. The final probe also used explicit `127.0.0.1` runtime addresses for optional local services; these overrides were not persisted.

Checks used separate runtime paths below `.venv/` so test state did not populate the default user-data databases. Temporary validation servers are shut down after the probe.

## Execution and Ownership

```text
main.py
  app.viewer.AIModelViewer
    app.startup_viewer.AIModelViewer
      tui_app.AIModelViewer
        Textual App

TUI search input
  debounce / active search generation / fresh search cache
    Textual background search worker
      search.SearchOrchestrator
        provider fan-out -> SearchResult -> SearchOutcome
      apply results / errors / pagination / cache
    app search state -> results helpers -> DataTable

TUI download actions
  app.download_manager
    downloads.service_client -> localhost HTTP
      downloads.api -> downloads.store (SQLite)
        downloads.download_service worker pool
          downloads.runner -> Ollama / Hugging Face download process
```

The runtime subclass chain matters: asynchronous startup replaces the synchronous startup in the base TUI, and the runtime viewer adds selectors, responsive modal/deletion behavior, live-first stale fallback, and table-content signatures. Editing the base class alone can miss the behavior used by `main.py`.

CLI and REST share providers, hardware, and scoring with the TUI but have their own search call paths. They do not inherit the TUI's search-cache fallback. REST model discovery currently supports Ollama and Hugging Face; optional provider availability metadata does not imply REST search support.

`search/search_cache.py` retains search pages only in memory. `core/cache_db.py` persists model metadata and hardware snapshots. `downloads/store.py` persists jobs independently of the TUI's lifetime. These are three separate state owners.

The download-only table update path is already implemented. `app/viewer.py` guards it with ordered rendered-content and column-layout signatures; changed content/order/layout takes the full rebuild path. Remaining performance work should measure these paths before expanding incremental updates.

## Limits and Next Work

- Ollama, LM Studio, Docker Model Runner, and MLX were not detected as available during the normal runtime probe. Actual execution against those runtimes is not validated here.
- Smoke checks use bounded startup modes; their success alone does not prove remote search or a model download.
- The normal `/api/v1/providers` call exceeded an initial 10-second client timeout. It performs provider detection synchronously; investigate repeated detection and HTTP retry/connect costs before changing timeout or availability contracts.
- No model weights were downloaded during validation.
- Treat `.planning/roadmap.md` as a priority list that still needs comparison with later merged fixes; do not reimplement table consistency, asynchronous startup/deletion, or live-first fallback.

To launch the installed TUI from the repository in PowerShell:

```powershell
& .\.venv\Scripts\python.exe main.py
```

To rerun the baseline checks:

```powershell
& .\.venv\Scripts\python.exe scripts/dev.py verify
& .\.venv\Scripts\python.exe scripts/dev.py smoke
```

Normal runs use the configured OS user-data directories. Validation path overrides were scoped to the checking processes, not saved to `.env`.

The machine-specific CA bundle, validation script, JSON report, logs, and SVG screenshot are under `.venv/runtime-probe/` (the script itself is `.venv/runtime_probe.py`). These are disposable local evidence, not committed project dependencies. Normal remote access in this environment may require the same trusted-CA configuration; no certificate bypass was added to production code.
