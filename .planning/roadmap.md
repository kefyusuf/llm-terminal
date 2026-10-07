# AI Model Explorer — Active Roadmap

**Baseline date:** 2026-10-02

**Baseline revision:** `32fe324b97294d32b73795b06a12158e85abe283`
**Status:** research-informed local-product release roadmap; release gates not yet complete

This roadmap retains the completed hardening baseline and replaces the older active backlog with priorities derived from the 2026-10-02 industry/release assessment. See [research and comparison](../docs/industry-readiness.md) and [local runtime evidence](../docs/runtime-validation.md). Completed work is not an active backlog.

## Implementation and remaining acceptance

Reviewed on 2026-10-05. Integration PR #143 delivered the roadmap and
implementation PRs #118–#142 onto main; see [continuation](continuation.md) for exact source
and evidence. This is not a publication or inference-readiness claim.

| Contract | Implemented/verified | Remaining work |
|---|---|---|
| M0 R1/R2/R3 | Nonblocking discovery/hardware startup, measured startup, doctor, transparent estimates and installed wheel/sdist entrypoints | Requalify the integrated release/version; prepared optional-runtime UX evidence |
| M1 D1 | Pinned selected artifact, public metadata, server-owned destination/disk plans and final byte/digest checks | Unknown upstream facts and license permission remain explicit |
| M1 D2/P1 | Windows/Linux HF cancellation/retry, Windows service stop/child exit, managed removal; native Windows tinyllama Ollama cancel/reopen/retry and blob hashes; CPU and two bounded fresh-server CUDA calibration runs | Original transient CUDA timeout cause unresolved; other native inference/backend/model evidence remains open |
| M1 O1 | Recorded catalog-source ADR, fixture-backed parsing and actionable drift diagnostics | No catalog migration without a verified supported replacement |
| M2 S1/A1/U1 | Saved runtime memory scenarios, bounded calibration tool, versioned exports/offline comparison and measured incremental table refresh | Genuine repeated local inference corpus before accuracy claims or scoring changes |
| M3 L1 | Build-once candidate pair, source/byte manifest and 18 isolated consumers | Authorized new version, publisher/account/name, protected environment, TestPyPI/attestation and promotion |
| M3 L2 | Security/support route, terminal-job backup and isolated previous-worker/file restore rehearsal | Real participant pilot; requalify actual release/rollback candidate and authorize promotion |

The milestone definitions below preserve the acceptance contracts. Implemented
items in this table are not new feature TODOs; only the remaining evidence and
release gates form the active work.

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

- Later merged work also includes asynchronous runtime service startup/deletion, narrow-terminal/modal fixes, table-content consistency guards, REST response hardening, packaging/lock fixes, and a staged mypy gate.
- Local 2026-10-02 evidence: 682 tests plus imports/Ruff, the scoped 32-file mypy check, all smoke surfaces, actual hardware detection, normal REST/service transport, and two real HF results rendered in the runtime TUI. HTTPS required a temporary Windows trusted-CA bundle; no TLS bypass was used.
- Coverage figures above are prior canonical CI evidence, not a new measurement on this Windows run.

## Product and Release Scope

First supported public product: a local terminal application for discovering, comparing, selecting and downloading exact model artifacts across runtimes, with scriptable CLI and localhost REST. Distribution target: versioned Python package with pipx instructions. A hosted multi-user service is a separate decision.

Differentiation hypothesis: trustworthy hardware-aware selection across providers, transparent estimates, keyboard workflow, and reliable acquisition. Validate it with pilot users. Chat/RAG/agents, billing, team administration and an inference engine are deferred.

Release dependency order: M0 -> M1 -> M3 pilot/promotion. M3 artifact/installation work starts in M0 and proceeds alongside M1 after the baseline is established. M2 calibration/automation is subsequent product work and can ship after a restricted prerelease; it is not required to add every feature before gathering user feedback.

## M0 — Accurate Product Contract and Runnable Candidate

### R1. Measure and bound startup/provider detection

- Trace detection in viewer construction, startup and REST provider listing; measure repetitions with optional runtimes absent and present.
- Avoid repeated synchronous network detection on the UI/request path; represent pending/unavailable/failed states honestly and bound cached snapshots.
- Preserve capability-authoritative behavior, cancellation and visible failure semantics.
- Provisional local targets: first usable TUI frame p95 <= 2s and input response <= 100ms while probes run; provider-list response <= 2s using cached/pending availability. Collect a named-machine baseline with at least 20 runs before accepting or revising these budgets. These are local UX targets, not upstream API SLAs.

Exit: measured before/after evidence plus focused absence/timeout/recovery tests; slow provider detection cannot prevent search input use.

### R2. Explain estimates and provide network diagnostics

- Label quality, throughput, fit and fallback context as estimates with assumptions/source; preserve existing public scoring fields until additive metadata is available.
- Do not describe model-size-derived context score as measured context capacity or parameter-count quality as benchmark quality.
- Add a user-invoked doctor/diagnostic command for runtime reachability, versions, model paths, free disk, TLS/custom CA and configuration. Redact tokens and sensitive paths/headers in shareable output.
- Document REQUESTS_CA_BUNDLE/custom trust troubleshooting; keep certificate validation enabled. Make no machine-specific trust changes in package defaults.

Exit: estimates are visibly distinct from facts; missing runtime and CA failures produce actionable bounded diagnostics without exposing tokens. Documentation accurately describes what is inferred.

### R3. Prove installation outside the source tree

- Strengthen Package CI: fresh environment, wheel install, working directory outside checkout, no source-tree PYTHONPATH; verify imported modules resolve to installation paths.
- Verify both console entry points, installed TUI startup, REST and download-service subprocess launch; separately build/install sdist to prove source completeness.
- Include macOS packaging checks if it is claimed as supported; validate claimed Python 3.10-3.14 install/import compatibility or deliberately narrow support metadata/docs.
- Record dependency-resolution behavior as well as committed development locks; those locks do not constrain all users' package installations.

Exit: candidate wheel/sdist work without repository files and the supported installation instructions are tested.

## M1 — Reliable Model Acquisition and Explicit Support

### D1. Artifact identity, license and preflight

- Carry provider/repository, exact file/variant, resolved revision/digest where available, source/model-card/license links, size and metadata timestamp.
- Persist resolved identity on queued jobs. Model unknowns explicitly; do not infer permission from the application MIT license or a model's download count.
- Show destination and known required bytes before queueing; check disk budget and path containment. Never silently acquire every quantization variant.
- Current HF SDK 0.36.2 supports revision but not upstream's newer dry_run API: use supported metadata or a separately validated dependency change.

Exit: selected and downloaded artifacts have consistent immutable identity where supported; low-disk/unavailable-metadata behavior is explicit; license/source information survives across TUI/CLI/REST where those surfaces expose the model.

### D2. Real download, cancellation and recovery acceptance

- Reuse existing single-file HF download, worker pool, cancellation escalation, SQLite state and orphan recovery rather than rebuilding them.
- Run bounded, opt-in acquisition checks against a small pinned HF artifact and a small Ollama model on a prepared host with disk/time limits.
- Cover cancel, interrupted transfer, restart, retry, completion validation, duplicate jobs and safe deletion of managed versus shared files. Record which behavior belongs to the SDK and which is guaranteed by this application.
- Exercise TUI/service independence and record the downloaded location plus runtime handoff instructions.

Exit: actual HF/Ollama success and interruption/recovery evidence, focused regressions for newly changed boundaries, and no false completed status for absent/invalid artifacts. Stable release requires this evidence; existing mocked tests alone do not satisfy it.

### P1. Publish an evidence-based compatibility matrix

- Track OS, Python, GPU/backend, provider/runtime version and supported operation: discovery, installed-list, download, runtime handoff.
- Validate Windows native and Linux native for the first supported release. Add WSL/macOS Apple Silicon/MLX evidence when prepared environments are available; mark unvalidated combinations experimental.
- Distinguish hosted bootstrap/verify/smoke from actual GPU and runtime acceptance. Do not block a narrowly scoped prerelease on every optional runtime.

Exit: every advertised supported combination has recorded evidence; unavailable environments remain an explicit gap rather than assumed success.

### O1. Keep Ollama catalog discovery resilient

- Retain fixture-backed parsing and observable shape failures.
- The researched official index exposes local model management; no supported global catalog-search replacement was established. Record an architecture decision before changing the remote discovery source.
- Use local structured model details for enrichment where available; do not confuse local tags with a global registry API.

Exit: a documented source decision and actionable drift diagnostics. Migration is conditional on a verified supported source, not a release prerequisite.

## M2 — Better Selection and Automation

### S1. Metadata-aware hardware planning and calibration

- Prefer authoritative context/architecture/quantization/capability data, retaining explicit fallback provenance.
- Model context/KV-cache, offload and concurrency assumptions; distinguish supported context from requested context and runtime allocation.
- Add a bounded opt-in benchmark/calibration path with hardware/runtime/model revision, workload, repetitions and dispersion. Separate prompt throughput, generation throughput and end-to-end latency.
- Publish prediction error on measured samples before promising accuracy. No universal model-quality ranking from file size.

Exit: fact/estimate/measurement are distinguishable; a repeatable calibration corpus evaluates prediction error without hiding unsupported cases.

### A1. Scriptable output and runtime handoff

- Add versioned additive JSON/export contracts for search/plan/comparison where users need them; preserve existing recommend stdout and diagnostics contracts.
- Include exact artifact identity, score provenance and errors; do not require UI scraping.
- Provide supported handoff/import instructions; converge REST/provider scope only for a demonstrated user workflow.

Exit: schema/compatibility tests and documented examples work from the installed package.

### U1. Measure remaining table costs

- Download-only incremental updates and render-signature guards already exist.
- Benchmark representative counts (e.g. 100/1,000 rows), structural versus download-only refresh, cursor/scroll and narrow layouts.
- Expand incremental updates only after evidence shows a material cost, retaining full rebuilds for layout/order changes where safer.

Exit: measured benefit plus selection/order/content regressions; no generic TUI rewrite.

## M3 — Controlled Public Release and Maintenance

### L1. Build-once candidate and publishing pipeline

- Produce versioned wheel/sdist, metadata checks, hashes and provenance from the candidate revision; preserve artifacts for install tests and publication.
- Rehearse a prerelease on TestPyPI with controlled dependency sourcing. Configure protected publishing environments and PyPI OIDC Trusted Publishing; capture attestations.
- Existing package workflow builds/tests but does not publish. External publisher/account/project-name availability and current remote CI remain prerequisites to verify during release work.

Exit: tested artifacts match publish inputs; exact-source applicable CI is green; trusted publisher/environment setup and release authorization exist before publication. This roadmap does not authorize a deployment by itself.

### L2. Pilot, promotion and recovery

- Give a small pilot group the actual package installation instructions. Track installation, model selection, download and recovery outcomes; collect only user-submitted redacted diagnostics.
- Publish tested/experimental support, known limits, versioned release notes and upgrade/uninstall instructions.
- Establish job-data backup/migration compatibility, previous-version installation and a hotfix/withdrawal procedure. PyPI yanking is not automatic client rollback; exact pins can still install a yanked version.
- Preserve existing daily external search checks, make provider-drift failures actionable, and distinguish upstream outages from deterministic merge failures.
- Define an issue/security-reporting route, dependency maintenance owner and a release evidence record.

Exit: pilot blockers resolved, M0/M1 mandatory gates passed, rollback/data compatibility exercised and release notes describe actual support. Stable promotion is based on evidence rather than a target calendar date.

## Release Decision

A restricted prerelease may expose experimental platforms/providers honestly. Stable public distribution requires clean installation, usable startup, transparent estimates, actual HF/Ollama acquisition and recovery evidence, safe managed-file deletion, declared compatibility and an operational release/upgrade process.

Historical 67.44% coverage is not new exact-head proof. Keep the 60% gate and add focused tests as consequential seams change; do not create percentage-only work or weaken exclusions/assertions. Remaining platform/runtime edge cases enter the relevant milestone only when their promised operation needs them.

## Documentation Sync Policy

Follow docs/maintenance.md. Update README, CHANGELOG, this roadmap and affected codebase references when behavior changes; record release evidence against the exact source/artifact. Distinguish researched recommendation, implemented behavior, local test evidence and remote release evidence.
