# Changelog

## Unreleased

- Updated Hugging Face model search to use `list_models(..., expand=["likes", "siblings"])`.
- Removed per-model repository file listing and now select GGUF candidates from expanded sibling metadata.

## 0.1.0 - 2026-02-26

- Added baseline tests and project documentation.
- Replaced broad exception handling with explicit error paths and user-facing search error state.
- Fixed stale threaded search updates and Hugging Face deduplication by full model ID.
- Improved Hugging Face search performance with caching and capped detailed metadata lookups.
- Refactored the application into modules: `app`, `hardware`, `providers`, and `utils`.
- Added explicit loading, empty-result, and error status messaging in the UI.
- Pinned dependency ranges and split runtime/dev requirements.
- Added live platform integration tests and a release preflight script.
