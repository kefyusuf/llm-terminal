# AI Model Explorer 1.0.1

## Highlights

- Major UI responsiveness improvements by moving heavy polling and service requests off the main UI thread.
- Modal stability hardening to prevent `NoMatches`/widget-race crashes during screen transitions.
- New keyboard-first compact workflow with runtime view toggle (`v`) and compact controls (`p`, `u`, `s`, `f`, `h`, `[`, `]`).
- Results table layout rebalanced for better horizontal usage and reduced dead space on medium/large terminals.

## Stability and Performance

- Added debounced resize reflow and reduced redundant table redraw pressure.
- Removed expensive side effects from status updates.
- Added async polling paths for system metrics and download-service endpoints (`/jobs`, `/debug/active`, `/health`).
- Added modal-aware UI polling pause/resume without interrupting background download-service jobs.

## UI and UX

- Added `compact` and `comfortable` view modes (`AIMODEL_UI_MODE`, runtime toggle with `v`).
- Compact mode now supports full filtering/sorting from keyboard:
  - Provider: `p`
  - Use Case: `u`
  - Sort: `s`
  - Fit: `f`
  - Hidden gems: `h`
  - Pagination: `[` and `]`
- Added compact chip status bar and color-segmented metric rendering.
- Merged top-row controls (search + provider + use-case) for denser workflows.

## Architecture and Maintainability

- Extracted multiple focused modules from the previous `app.py` monolith:
  - `search_cache.py`
  - `search_orchestration.py`
  - `download_history.py`
  - `download_lifecycle.py`
  - `download_status.py`
  - `results_layout.py`
  - `results_view.py`
  - `results_presenter.py`
  - `results_text.py`
  - `release_check_helpers.py`
- Expanded unit coverage for extracted helpers and compact-mode behavior.

## Compatibility Notes

- No breaking user-facing commands.
- Background download jobs remain managed by the download service and are not paused by modal UI behavior.

## Validation

- `pytest -q` passing.
- `python scripts/release_check.py` passing (`Hugging Face` and `Ollama` live checks).
