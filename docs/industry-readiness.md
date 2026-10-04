# Industry Comparison and Release Readiness

Research date: 2026-10-02. Local source baseline: `32fe324`.

This assessment combines official product/distribution documentation with the current repository and the [runtime validation](runtime-validation.md). Findings about other products describe their documented scope, not comparative performance measurements. Priorities and acceptance targets below are project decisions inferred from the evidence, not universal industry requirements.

## Product Position

AI Model Explorer should focus on developers selecting and acquiring local models across runtimes: discover -> understand compatibility -> compare -> select an exact artifact -> download -> hand off to a runtime.

The proposed first public release is an installable local terminal application with CLI and localhost REST support. A hosted multi-user service is a separate product decision. Runtime integration does not require building another inference engine, chat application, RAG platform, or model registry.

| Reference | Documented scope | Implication for this project |
|---|---|---|
| [LM Studio model discovery](https://lmstudio.ai/docs/app/basics/download-model) | Search supported models, choose quantization variants, configure a model directory | Exact variant selection and a clear acquisition path are baseline user needs |
| [LM Studio getting started](https://lmstudio.ai/docs/app/basics) | Download, load, then use a model | Download success should lead to clear runtime handoff instructions |
| [Ollama model details](https://docs.ollama.com/api-reference/show-model-details) | Local structured metadata includes license, capabilities, quantization, and model information | Prefer authoritative local metadata where available; retain provenance |
| [Docker Model Runner](https://docs.docker.com/ai/model-runner/) | Pull/run/serve models and package model artifacts for OCI registries | Integrate with lifecycle/capabilities rather than duplicating the runtime |
| [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards) | Intended use, limitations, license, and evaluation metadata | Popularity and a heuristic score cannot replace source/evaluation information |
| [llama.cpp benchmark tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) | Repeated performance measurements with configurable workloads and structured output | Keep predicted throughput separate from measured benchmark evidence |

The differentiation hypothesis is transparent hardware-aware selection across providers, a fast keyboard workflow, and reproducible model acquisition. This is a hypothesis to validate with pilot users, not a claim that market demand has already been established.

## Current System Compared with Required User Outcomes

| User outcome | Current evidence | Gap and decision | Milestone |
|---|---|---|---|
| Install without cloning the repository | Packaging metadata, console entry points, Windows/Linux wheel CI | Package checks run from the checkout; validate installed commands/imports from an unrelated directory and fresh environment | M0 |
| Understand why a model is recommended | Four scoring dimensions, hardware/MoE/quantization estimates | Quality is a heuristic; context score is derived from model size in `compute_context_score()`. Label estimates and show assumptions/unknown metadata; do not present them as measured quality/context | M0/M2 |
| Start despite unavailable providers | Structured failures, asynchronous service startup, optional-provider containment | Runtime probe hit a >10s provider-list timeout; constructors and REST provider listing repeat synchronous detection. Measure and bound first paint/detection/request latency | M0 |
| Diagnose network/certificate problems | Errors are surfaced; live HF search worked after a temporary trusted-CA override | Provide documented custom-CA support and useful diagnostics; preserve TLS verification and avoid exporting machine trust as a universal package fix | M0 |
| Select the intended model artifact | HF commands carry exact repository and filename; regression coverage exists | Persist resolved revision/digest, source metadata, size and artifact identity; handle missing information explicitly | M1 |
| Download reliably under interruption | SQLite jobs, bounded workers, cancellation escalation, orphan recovery and migrations exist | Need real small-artifact evidence for cancel/retry/restart, low disk, integrity/completion, and deletion safety; distinguish SDK behavior from application guarantees | M1 |
| See model restrictions and provenance | Publisher/source/basic model information available | Canonical license/model-card links, revision and metadata freshness are not carried as a complete user-facing contract | M1 |
| Use the downloaded model | Runtime detection and installed-model markers; download paths for Ollama/HF | Verify runtime-specific handoff; validate at least one real Ollama pull and one HF acquisition before claiming end-to-end support | M1 |
| Automate without scraping UI text | CLI and REST exist; recommend JSON and structured REST errors exist | Search/plan/export JSON needs a versioned additive contract; downloads are a separate service surface | M2 |
| Rely on advertised platform support | Hosted Windows/Linux and macOS lanes; Python metadata supports 3.10-3.14 | Hosted smoke does not validate real GPU/runtime combinations, WSL, or Apple Silicon MLX; declare tested versus experimental support | M1/M3 |
| Upgrade and recover safely | Persistent job state and legacy migrations exist | No documented release rollback, upgrade-data backup/compatibility, or bad-release withdrawal procedure | M3 |

These gaps were checked against `core/scoring.py`, `core/models.py`, providers, `downloads/runner.py`, `downloads/store.py`, the package/live workflows, and the preceding runtime report. A missing application contract does not imply the underlying provider SDK lacks the capability.

## Model Data and Recommendation Contract

Use a canonical identity that retains provider, repository/model name, selected file or variant, and resolved immutable revision/digest where the provider exposes one. Display license/source links, task capabilities, format, quantization, download size, supported context, metadata source, and retrieval time. Missing fields stay unknown rather than being inferred silently.

Separate three types of evidence: provider-reported metadata, application estimates, and actual measurements. Preserve current scoring compatibility while adding provenance and confidence fields. Context capacity and estimated memory required at a chosen context are different values. [Ollama documents increased memory use at larger context lengths](https://docs.ollama.com/context-length); this justifies a metadata-aware KV-cache/offload model, not a universal guarantee derived from weight size.

Performance calibration should record hardware, runtime/version, model revision, quantization, prompt/decode workloads, context, offload, repetitions and dispersion. [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) is one suitable reference; its measurements exclude tokenization and sampling, so they must not be labeled full end-to-end latency.

For Ollama remote discovery, the [official documentation index](https://docs.ollama.com/llms.txt) lists local model-management endpoints. This research did not establish a documented global registry-search endpoint. Local `/api/tags`/`/api/show` must not be treated as a replacement for remote catalog search. Retain observable fixture-backed HTML discovery until a supported alternative is verified.

## Download and State Contract

Before queueing, show exact artifact, destination, expected bytes if known, and available disk space. Persist the resolved identity with the job; define what retry and restart mean for partial data. Completion needs an accessible expected artifact and provider-supported size/digest evidence where available; unknown verification stays explicit. Destructive file operations must stay inside the configured managed model directory and must not remove shared artifacts accidentally.

[Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/download) supports revision selection and selective acquisition. Current upstream documentation also exposes dry-run APIs, but this repository's installed 0.36.2 signatures have no `dry_run` parameter. Implement preflight through supported metadata APIs or a separately tested dependency upgrade; do not copy current upstream examples into this pinned environment blindly. SDK cache/partial-download behavior is a dependency capability that still needs application-level recovery tests.

## Public Release Path

Use a versioned Python package as the first distribution channel. [PyPA documents pipx isolation](https://packaging.python.org/en/latest/guides/installing-stand-alone-command-line-tools/), which fits a terminal application. Native installers can follow demonstrated user demand; they are not a prerequisite for the first supported developer release.

1. Complete M0/M1 exit criteria and record the exact release candidate commit. Keep deterministic CI separate from scheduled external-service checks.
2. Build wheel/sdist once, validate distribution metadata, record checksums, and run installation/entry-point/TUI/API/service checks from outside the checkout. Also build/install the sdist independently to verify source completeness.
3. Exercise the tested artifacts on a narrowly declared Windows/Linux support matrix; publish experimental labels for unvalidated combinations. Validate claimed Python versions or narrow the claim deliberately.
4. Prepare a prerelease and TestPyPI rehearsal. Install the candidate with controlled dependency sourcing; do not assume TestPyPI contains all runtime dependencies.
5. Publish the already tested artifacts through a protected release workflow using [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) and [distribution attestations](https://docs.pypi.org/attestations/). The [PyPA release guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) provides the build/artifact-transfer/publish pattern. Actual publisher/environment configuration remains an external prerequisite, not work completed by editing YAML.
6. Run a small pilot using the actual installation instructions. Record installation, discovery, selection, download and recovery failures through an issue template and a user-invoked redacted diagnostic bundle. No automatic usage upload is needed for this release.
7. Promote the same validated artifact set to stable when release gates pass. Publish support boundaries, known limitations, changelog, hashes, and upgrade/uninstall instructions.
8. If a release is broken, document explicit previous-version installation, release-data compatibility and a hotfix. [PyPI yanking](https://docs.pypi.org/project-management/yanking/) discourages normal selection of bad versions but is not client rollback and can still permit exact pins. Do not promise deletion/replacement of an already published version.

Current evidence demonstrates local runtime behavior, not a completed public release: no registry account/configuration, existing PyPI listing, remote branch freshness, or current hosted CI result was verified here. This assessment does not publish, tag, or deploy anything.

## Release Gate

Stable release is blocked until the candidate artifact has clean-machine installation evidence, deterministic checks at its source revision, real HF/Ollama acquisition evidence, interruption/restart/delete safety evidence, explicit estimate/provenance presentation, documented trusted-CA troubleshooting, and an operational upgrade/withdrawal process. A restricted prerelease can carry clearly declared experimental platform/provider support.

New UX latency targets in the roadmap are provisional budgets requiring a benchmark baseline. Raising aggregate coverage alone, adding more providers, or rewriting the TUI is not an exit criterion. Preserve the existing 60% coverage gate and add focused regressions for consequential changes.

## Deferred Scope

Hosted multi-user APIs, team RBAC/billing, central model registries, chat/RAG/agents, autonomous inference routing, and a plugin marketplace require separate demand and operational design. Backend libraries may support them; their existence is not evidence that this product needs them now.

The ordered implementation backlog and milestone exit criteria are in [the active roadmap](../.planning/roadmap.md).
