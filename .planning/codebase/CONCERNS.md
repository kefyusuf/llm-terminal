# Codebase Concerns

**Baseline date:** 2026-09-07  
**Baseline revision:** `d2c8913f3623eaaab8087a3c9d427b0a2686b375`

This file lists **current, verified concerns**. Resolved mapper findings are retained only in the historical summary so completed work is not accidentally re-planned.

## Priority 1

### 1. Ollama remote registry depends on HTML structure

**Area:** `providers/ollama_provider.py`

Remote Ollama discovery still fetches `ollama.com/search`/library HTML and parses it with BeautifulSoup.

**Risk:**

- third-party page structure can change without a versioned API contract,
- future search or model-detail page shapes can drift beyond the sanitized fixture corpus,
- metadata/pagination behavior is constrained by page representation.

**Current mitigation:**

- retry/backoff and provider diagnostics exist,
- sanitized fixture-backed contracts pin the currently supported search-anchor and model-detail table/card shapes,
- deterministic tests cover ordering, dedupe, ignored links, direct/parent pull counts, GB/MB size parsing, preferred variants, genuine zero results, and unsupported search HTML,
- search HTML with recognized models follows the existing success path; a resultless page containing the verified `No models found.` marker is a clean zero result,
- a resultless HTTP-200 search page without that marker emits the stable legacy diagnostic plus a non-retryable structured `parse_error` instead of silently reporting success,
- nested pull-count text boundaries are regression-covered after the #86 correction.

**Next work:**

- research a supported structured registry/search source before considering migration,
- keep model-detail unsupported-shape policy separate until there is concrete evidence/caller impact that justifies expanding the structural-failure boundary.

### 2. Aggregate coverage is healthy but uneven across high-risk modules

**Area:** TUI, download API, platform hardware probes

The canonical Ubuntu/Python 3.12 coverage lane now measures **67.44% total coverage** (`5095` statements, `1659` missed) and enforces a **60%** floor.

Focused deterministic tests have removed several consequential weak points:

- `downloads/runner.py` — **24% → 90%**,
- `downloads/service_client.py` — **46% → 90%**,
- `core/hardware.py` — **44% → 52%**, with atomic NVIDIA probe-state coverage.

Remaining measured examples:

- `app/modals.py` — 25%,
- `tui_app.py` — 38%,
- `downloads/api.py` — 46%,
- `core/hardware.py` — 52%,
- `app/viewer.py` — 57%.

**Risk:**

- aggregate coverage can remain green while consequential UI/platform/API branches are weakly exercised,
- future changes in lower-coverage modules can regress without a focused contract test.

**Current mitigation:**

- 60% exact-head aggregate gate,
- 600+ deterministic tests,
- separate verify/smoke matrices,
- high coverage on provider/search/store/runner/service-client seams,
- regression-test requirement for correctness fixes.

**Next work:**

- add focused tests when concrete correctness/resilience/platform/performance work touches the remaining weak modules,
- do not create percentage-only PRs indefinitely,
- do not inflate coverage by excluding meaningful production modules or weakening assertions.

### 3. TUI result refresh still performs full table rebuilds in important paths

**Area:** `tui_app.py`, `results/`

Result refresh still uses `DataTable.clear()` in normal/structural refresh paths.

**Risk:**

- unnecessary work for larger result sets,
- cursor/scroll/update churn,
- download/progress state changes may trigger more rendering than necessary.

**Current mitigation:**

- resize/search progress is already debounced/throttled,
- cursor/scroll behavior has dedicated handling,
- pure layout/presentation logic is extracted and tested.

**Next work:**

- measure representative refresh cost,
- use stable row keys and incremental cell/row updates where ordering/layout does not change,
- keep full rebuilds where they are simpler/correct for structural changes.

## Priority 2

### 4. Platform acceptance is broader than hosted CI evidence

Project metadata supports Python 3.10-3.14 and integrations span Windows/Linux/macOS/Apple Silicon. Hosted CI verifies:

- Ubuntu + Windows on Python 3.12 + 3.14,
- offline-safe verify/smoke behavior on those environments,
- canonical aggregate coverage on Ubuntu/Python 3.12,
- macOS/Python 3.12 bootstrap + verify + smoke through the compatibility dependency wrapper.

The hosted macOS lane closes the basic installation/testability gap, but hosted CI still does not prove:

- WSL-specific behavior,
- real NVIDIA/AMD/Intel GPU tooling,
- actual Ollama/LM Studio/Docker Model Runner services,
- real Apple Silicon/MLX cache/filesystem/runtime behavior,
- native macOS lock reproducibility; macOS currently resolves through `requirements/requirements-dev.txt` until dedicated lock artifacts exist.

**Next work:** expand the automated/manual platform acceptance matrix where concrete platform behavior requires it, with particular focus on real Apple Silicon/MLX and local-provider integrations.

### 5. Main TUI base module remains large

The old “single root `app.py` monolith” concern is obsolete: provider selector, modals, widgets, download manager, search state, search orchestration, result helpers, and download internals have been extracted.

However, `tui_app.py` still owns the base Textual layout/event lifecycle and remains a high-coupling presentation module.

**Risk:** broad UI changes can still have regression surface across search, table rendering, polling and downloads.

**Guidance:**

- prefer pure helpers/delegates for new deterministic logic,
- avoid extraction solely to reduce line count,
- require focused Textual tests for behavior-changing edits.

## Priority 3

### 6. Local-provider helper parity has minor remaining edge cases

Known lower-priority examples:

- MLX `list_installed()` does not yet have the same explicit filesystem-I/O containment contract as MLX search,
- LM Studio metadata helper parsing is less hardened than the production search/list paths, but currently lacks a significant production caller,
- some zero/negative limit edge cases are unreachable through normal validated user surfaces.

These should be handled when a concrete caller/bug justifies them rather than bundled into speculative refactors.

### 7. Textual 0.x dependency pin

The project intentionally pins `textual>=0.86,<1.0`.

**Risk:** upgrading Textual can affect widget/layout/event behavior and requires TUI smoke/regression evidence.

**Guidance:** treat Textual upgrades as compatibility changes, not routine dependency bumps.

## Security Boundaries

### Download service

Current boundary is intentionally local:

- loopback-only host,
- optional bearer token on non-health endpoints,
- non-loopback hosts rejected until TLS exists.

Do not weaken this restriction as a convenience change. Any LAN/remote mode requires an explicit transport/auth threat model.

### REST API

The REST API binds to loopback by default and is intended for local programmatic use. It does not provide a public-network authentication model.

If remote binding ever becomes a product requirement, add auth/TLS/origin/rate-limit design before exposing it.

### Hugging Face credentials

Use least-privilege/read-only tokens. Credential propagation changes should be reviewed separately from unrelated provider correctness work.

## Resolved Historical Concerns

The following old concerns are retained here only to prevent accidental re-planning:

- **Root `app.py` 2466-line monolith:** obsolete; runtime is `main.py` → `app/viewer.py` → `tui_app.py` with extracted modules.
- **No HTTP pooling:** resolved by `core/http_client.py` shared session.
- **No provider retry/backoff:** resolved for retryable provider HTTP paths.
- **No formal provider errors:** resolved by `ProviderError` + `SearchResult.structured_errors`.
- **Sequential provider search:** resolved by `SearchOrchestrator` parallel fan-out.
- **No search cancellation:** resolved by orchestrator cancellation polling/UI search IDs.
- **Provider registry silent suppression:** lazy imports now contain only expected `ImportError` with warnings; unexpected import-time programming failures propagate, and unexpected optional-provider detection failures fail closed with warnings.
- **Provider-selector broad synchronization catch:** narrowed to expected Textual `NoMatches`; other synchronization failures propagate.
- **Download polling/sync silent stale-state paths:** expected service failures preserve last-known state with diagnostics and unexpected programming failures propagate.
- **Residual download action broad catches:** classified as intentional user-visible containment; start/cancel/delete operations return explicit failure status rather than false success.
- **Hardware broad-probe audit:** AMD/Apple/Intel use bounded platform exception sets; NVIDIA NVML broad catches remain intentional third-party backend fallbacks, with state committed atomically only after a complete probe so partial failures cannot create false CUDA state.
- **Residual silent-failure audit:** completed for the tracked provider registry, selector, download polling/sync/action, and hardware-probe boundaries.
- **SQLite connection per operation:** resolved for metadata cache by shared locked connection.
- **Download service single worker:** resolved by bounded configurable worker pool.
- **Duplicate model-size estimator:** `core.utils` delegates to canonical MoE-aware estimator.
- **GPU bandwidth repeated linear scan:** resolved by pre-built lookup maps.
- **No REST input validation:** core model-search/plan inputs now validate to 400 responses.
- **Provider failures hidden in REST/CLI:** resolved with REST diagnostics and CLI stderr warnings.
- **Coverage configured but not enforced:** resolved by a canonical Ubuntu/Python 3.12 CI coverage job.
- **Staged aggregate coverage goal:** completed at **50% → 55% → 60%**, with current measured aggregate **67.44%**.
- **Download runner execution largely untested:** resolved with deterministic fake-process/state coverage raising `downloads/runner.py` from 24% to 90%.
- **Download service-client lifecycle largely untested:** resolved with deterministic lifecycle/request tests raising `downloads/service_client.py` from 46% to 90%.
- **Legacy `shell=True` finding:** no active source match; old planning reference removed from current concerns.

## Maintenance Rule

A concern should remain in this file only while it is materially true. When a merge resolves or substantially changes a concern, update/remove it in that PR when practical. See `docs/maintenance.md`.