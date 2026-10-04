# AI Model Explorer — Active Roadmap

Release-readiness work is proposed in PR #118. Consult the
[verified continuation record](continuation.md) for the current unmerged
implementation stack, exact-source evidence and remaining gates; the historical
baseline below is not a claim that those later PRs have merged.

**Baseline date:** 2026-09-07  
**Baseline revision:** `d2c8913f3623eaaab8087a3c9d427b0a2686b375`  
**Status:** post-hardening baseline; active roadmap only

This roadmap replaces the 2026-06-09 mapper plan. Completed work stays documented here only as a baseline summary; it is not an active backlog.

## Delivery Principles

1. Prefer small, reversible PRs with a single acceptance contract.
2. Treat exact-head CI evidence as the merge gate. A new head SHA invalidates prior green evidence.
3. Preserve existing public contracts unless the PR explicitly changes them.
4. Add focused regression coverage for every correctness fix before merge.
5. Keep provider failures observable: no silent false-success paths.
6. Review documentation after every successful merge. Update affected docs in the same PR when practical; otherwise open an immediate narrow docs follow-up before unrelated feature work.
7. Raise quality gates only when the repository already passes them; do not weaken tests or exclusions to make a gate green.

## Completed Baseline

The following items from the old roadmap are already implemented and should not be re-opened as generic tasks:

- Modular TUI support modules under `app/` for viewer extensions, modals, widgets, download management, search state, and constants.
- Parallel provider fan-out in `SearchOrchestrator` with bounded workers and cancellation polling.
- Provider capability metadata as the authority for search, pagination, installed-list, and download behavior.
- Shared `requests.Session` pooling plus retry/backoff for retryable GET failures.
- Structured provider diagnostics via `ProviderError`, preserved through provider results and orchestration.
- Structured/legacy diagnostic containment for Hugging Face, Ollama, LM Studio, Docker Model Runner, and MLX search paths.
- Provider registry lazy imports contain expected `ImportError` with warnings, unexpected import-time programming failures propagate, and unexpected optional-provider detection failures fail closed with warnings.
- Provider-selector widget synchronization contains only expected Textual `NoMatches` lifecycle absence; other synchronization/programming failures propagate.
- REST `/api/v1/models` additive `errors` and `structured_errors` output.
- CLI provider diagnostics on stderr while preserving script-safe JSON stdout.
- Search cancellation and provider-authoritative pagination handling.
- Shared SQLite cache connection with serialized access and retry-on-connection-error behavior.
- Bounded multi-worker download service (`AIMODEL_DOWNLOAD_MAX_WORKERS`, default `2`).
- Loopback-only download-service transport with optional bearer-token protection for non-health endpoints.
- Periodic TUI download polling preserves last-known state on expected service failures, surfaces deduplicated stale/recovery status, and no longer swallows unexpected programming failures.
- Action-triggered download synchronization preserves last-known state on expected service failures, reports the failed refresh, and no longer swallows unexpected programming failures.
- Download start/cancel/delete broad catches are classified as intentional user-visible containment: callers receive explicit failure status rather than false success.
- Hardware probe fallbacks are classified as intentional best-effort platform detection; NVIDIA NVML state is committed only after a complete probe so partial failures cannot leave false CUDA state or suppress later vendor fallbacks.
- Platform-specific user-data locations for cache/download state and Hugging Face model files.
- MoE-aware model-size estimation delegated to one canonical implementation.
- Pre-built GPU bandwidth lookup instead of repeated linear scans.
- REST parameter validation for provider, limit, context, and sort inputs.
- CI verify + smoke matrices on Ubuntu and Windows with Python 3.12 and 3.14.
- More than 600 deterministic tests in the current verify lane.
- TUI stale-search-cache fallback for disconnected/offline search recovery.
- Canonical coverage measurement lane on Ubuntu/Python 3.12.
- Staged aggregate coverage gate ratchet **50% → 55% → 60%**.
- Focused fake-process/state coverage for `downloads/runner.py`, raising that module from 24% to 90%.
- Focused lifecycle/request coverage for `downloads/service_client.py`, raising that module from 46% to 90%.
- Focused NVIDIA probe coverage raised `core/hardware.py` from 44% to 52% while pinning atomic state semantics.
- Current canonical coverage evidence: `5095` statements, `1659` missed, **67.44%** aggregate, **60% enforced floor**.
- Residual silent-failure audit completed for the previously tracked provider registry, selector synchronization, download polling/sync/action, and hardware-probe boundaries.
- Sanitized fixture-backed Ollama parser contracts pin supported search-anchor and model-detail table/card shapes, including ordering, dedupe, filtering, pull counts, size parsing, preferred variants, and genuine zero-result behavior.
- Ollama search structural-failure detection distinguishes the verified `No models found.` zero-result marker from unsupported HTTP-200 page shapes and emits aligned legacy plus non-retryable structured `parse_error` diagnostics instead of silent false success.

## P1 — Quality Baseline

### Q1. Targeted coverage when consequential weak seams are touched

The staged aggregate coverage goal and the residual silent-failure audit are complete. Coverage remains a merge gate and should continue to improve when concrete work touches weak boundaries; there is no active generic “raise the percentage” or “remove every broad catch” task.

Remaining lower-coverage modules include:

- `app/modals.py` — 25%,
- `tui_app.py` — 38%,
- `downloads/api.py` — 46%,
- `core/hardware.py` — 52%,
- `app/viewer.py` — 57%.

Do not create percentage-only PRs indefinitely. Add focused tests when these modules are changed for a concrete correctness, resilience, performance, or platform goal. Future aggregate gate increases should be evidence-driven and should not use new production exclusions or weaker assertions.

## P1 — Ollama Registry Resilience

Ollama remote discovery still depends on `ollama.com` HTML structure. Fixture-backed parser contracts and search structural-failure detection are complete; the remaining P1 work is evaluating a supported structured source that could reduce or replace HTML scraping.

### O3. Structured-source/API research

With deterministic fixture coverage and observable search-shape failure detection in place, research whether Ollama exposes a supported structured registry/search source suitable for replacing or reducing HTML scraping. Do not migrate until compatibility, pagination, metadata quality, and rate-limit behavior are understood.

## P2 — TUI Results Performance

The TUI still clears/rebuilds result table rows in important refresh paths.

### U1. Measure before changing

- Add a repeatable benchmark/test harness for row refresh at representative result counts.
- Separate column-layout rebuilds from ordinary row-state updates.

### U2. Incremental updates where safe

- Use stable row keys and `DataTable.update_cell()`/row updates for download/progress/state changes that do not require full re-sort/re-filter.
- Keep full rebuilds for structural column changes or result-order changes when simpler and safer.
- Preserve cursor/scroll position and compact/comfortable layout behavior.

## P2 — Platform Acceptance Matrix

CI currently verifies Ubuntu and Windows on Python 3.12/3.14. Runtime support remains Python 3.10-3.14, but provider/platform acceptance needs clearer evidence.

Build an explicit matrix for:

- Windows native,
- WSL/Linux,
- Linux native,
- macOS Apple Silicon.

For each platform record:

- bootstrap + package install,
- CLI smoke,
- REST smoke,
- TUI startup smoke,
- cache/download data paths,
- provider detection behavior,
- relevant runtime integrations (Ollama, LM Studio, Docker Model Runner, MLX).

Automate only where runners and services make the result reliable; keep clearly documented manual acceptance where hosted CI cannot represent the real runtime.

## P3 — Provider and Runtime Edge Cases

Address these only after P1 work unless a concrete user-facing defect is found:

- MLX `list_installed()` filesystem I/O containment parity with MLX search.
- LM Studio metadata helper response-shape containment if/when it has a production caller.
- Docker/LM Studio/MLX limit and installed-list edge cases not reachable through current user-facing limits.
- Further type tightening around provider duck-typed interfaces.

## P3 — Maintainability and Product Work

Candidate work after the quality/resilience baseline:

- More scriptable structured CLI output without breaking existing JSON contracts.
- REST/provider surface convergence where it materially improves users rather than adding abstraction for its own sake.
- Additional cache/offline behavior beyond the existing TUI stale-cache fallback, if CLI/REST offline workflows become a product requirement.
- Packaging/release automation improvements once release cadence requires them.

## Documentation Sync Policy

`docs/maintenance.md` is the operational rule for keeping documentation current.

At every successful merge, review at minimum:

- `README.md` for user-facing behavior/configuration,
- `CHANGELOG.md` for release-significant behavior,
- this roadmap for completed/changed priorities,
- `.planning/codebase/` when architecture, integrations, testing, stack, or known concerns changed.

A merged implementation is not considered fully documented if it leaves a known active document materially false.
