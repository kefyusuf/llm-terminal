# Changelog

## Unreleased

- Updated Hugging Face model search to use `list_models(..., expand=["likes", "siblings"])`.
- Removed per-model repository file listing and now select GGUF candidates from expanded sibling metadata.
- Deferred exact Hugging Face file-size lookup to model-detail open for faster search listing.
- Added short-TTL query-result cache in the app to speed repeated searches.
- Added hardware-aware cache invalidation when free RAM/VRAM changes beyond threshold.
- Added size-confidence indicator (estimated vs exact) in model details.
- Color-coded size-confidence indicator for quicker visual scanning.
- Added use-case filter controls (chat, coding, vision, reasoning, math, embedding, general).
- Added Hidden Gems filter for Hugging Face models with high downloads and relatively low likes.
- Added publisher column and source metadata to show sub-publishers such as `unsloth`.
- Added provider-specific rate-limit and HTTP error messaging for Hugging Face and Ollama registry.
- Surfaced startup hint when Ollama is not running and local runtime features are unavailable.
- Clarified Ollama requirement in project documentation.

## 0.1.0 - 2026-02-26

- Added baseline tests and project documentation.
- Replaced broad exception handling with explicit error paths and user-facing search error state.
- Fixed stale threaded search updates and Hugging Face deduplication by full model ID.
- Improved Hugging Face search performance with caching and capped detailed metadata lookups.
- Refactored the application into modules: `app`, `hardware`, `providers`, and `utils`.
- Added explicit loading, empty-result, and error status messaging in the UI.
- Pinned dependency ranges and split runtime/dev requirements.
- Added live platform integration tests and a release preflight script.
