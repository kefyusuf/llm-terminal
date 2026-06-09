# Coding Conventions

**Analysis Date:** 2026-06-09

## Naming Patterns

**Files:**
- `snake_case.py` — All Python files without exception
- `__init__.py` present in every package

**Functions:**
- `snake_case` — All functions and methods
- Internal helpers prefixed with `_` (e.g., `_select_preferred_gguf`, `_cpu_count`, `_row_to_dict`)
- Thread workers use `_run_..._worker` suffix pattern (e.g., `_run_system_info_refresh_worker`, `_run_download_poll_worker`, `run_search_worker`)

**Variables:**
- `snake_case` — All variables
- Module-level constants: `UPPER_SNAKE_CASE` (e.g., `SERVICE_VERSION = "1.7"`, `MIN_SERVICE_VERSION`, `GPU_BANDWIDTH`, `USE_CASE_WEIGHTS`)
- Class-level constants: `UPPER_SNAKE_CASE` (e.g., `RESULTS_COLUMN_LABELS_COMFORTABLE`, `USE_CASE_OPTIONS`, `SORT_OPTIONS`, `FIT_OPTIONS`)

**Types:**
- `PascalCase` — Classes: `AIModelViewer`, `HardwareMonitor`, `BaseProvider`, `ModelDetailModal`, `Settings`, `Scores`
- `PascalCase` — TypedDicts: `ModelResult`
- `PascalCase` — Dataclasses: `Theme`, `QuantInfo`, `Scores`

## Code Style

**Formatting:**
- Ruff formatter (line length 100)
- `target-version = "py310"`

**Linting:**
- Ruff linter with rules: E, W, F, I, UP, B, C4, SIM, RUF
- `E501` (line too long) explicitly ignored — handled by formatter
- Per-file exceptions for import ordering (`I001`) in many files via `[tool.ruff.lint.per-file-ignores]`
- Additional per-file exceptions: `RUF012`, `SIM105`, `SIM117`, `RUF010`, `E741`, `RUF001`

**Type Checking:**
- Mypy enabled but `strict = false`
- `ignore_missing_imports = true` — third-party packages not checked
- `warn_unused_ignores = true` — catch stale suppressions
- `# type: ignore[assignment]` used sparingly (e.g., `api_server.py:46` for shared state)
- `# noqa: E402` used for import-order violations after module-level code (e.g., `api_server.py:21-31`)

## Import Organization

**Order:**
1. Standard library (stdlib)
2. Third-party packages
3. Local/first-party imports (app, core, providers, downloads, results, search, scripts)

**Path Aliases:**
- No path aliases — all imports use relative package paths (e.g., `from .. import config`)
- Root-relative imports used from entry points (e.g., `from core.hardware import HardwareMonitor`)
- Lazy imports inside functions to avoid circular deps (`providers/__init__.py` has multiple lazy `_get_*_provider()` functions)

**Common import groups:**
```python
# Pattern: (1) stdlib, (2) third-party, (3) local
import json
import os
import sys

import requests
from textual.app import App, ComposeResult

import config
from core.hardware import HardwareMonitor
```

## Error Handling

**Patterns:**
1. **Broad `except Exception: pass`** — This is the dominant pattern. Used extensively to make the UI resilient to individual component failures. Examples:
   - `app.py:967` — `_apply_ui_mode()` queries UI widgets that may not exist
   - `app.py:1237` — Timer stop during resize
   - `app.py:1532` — Hardware refresh failures
   - `core/cache_db.py:72` — Database errors silently swallowed
   
2. **Error tuples** — Provider search returns `(results, errors)` tuple where errors is a `list[str]`
   
3. **Status bar messages** — User-facing errors go to `self.update_status()` at the bottom of the TUI

4. **Loguru** — Used in provider code for background errors; not used in `app.py`

5. **Graceful degradation** — When hardware detection fails, fallback specs are used

## Logging

**Framework:**
- loguru (`core/logging_.py`)
- Console: INFO level with colorization
- File: `data/logs/app_{time}.log`, DEBUG level, 10MB rotation, 7-day retention

**Patterns:**
```python
from core.logging_ import get_logger
logger = get_logger(__name__)
# or
from loguru import logger
logger.warning("Hugging Face search failed: {exc}", exc=exc)
```

## String Formatting

- **Primary:** f-strings (Python 3.10+)
- **Rich Markup:** All TUI display strings use Rich markup tags (e.g., `f"[bold #4fe08a]{text}[/bold #4fe08a]"`)
- **`str(exc)`** — Exception formatting uses `str(exc)` or `f"{exc}"` consistently

## Comments

**When to Comment:**
- Module-level docstrings on most files (e.g., `"""4-dimension scoring engine for LLM model evaluation."""`)
- Class docstrings on important classes (e.g., `HardwareMonitor`, `BaseProvider`)
- Method docstrings on public API methods (e.g., `estimate_tok_per_s()`, `score_model()`)
- Inline comments for non-obvious logic (e.g., `# Ollama doesn't support page-based pagination via URL`)
- ASCII art in `terminal_ui/app.py` for decorative header

**JSDoc/TSDoc:**
- Google-style docstrings (Args/Returns sections with types)
- Example:
```python
def estimate_tok_per_s(
    model_size_gb: float,
    gpu_name: str,
    mode: str,
    backend: str = "cuda",
) -> float:
    """Estimate tokens per second for a model on given hardware.

    Formula: ``tok/s = (bandwidth_GB/s / model_size_GB) x efficiency_factor``

    Args:
        model_size_gb: Model size in GB.
        gpu_name: GPU name string for bandwidth lookup.
        mode: Inference mode.
        backend: Backend type for fallback bandwidth.

    Returns:
        Estimated tokens per second, or ``0.0`` for No Fit.
    """
```

## Function Design

**Size:**
- Most functions are under 50 lines
- Exceptions: `_apply_ui_mode()` (68 lines), `refresh_table()` (~100 lines), `run_search_worker()` (~120 lines), `search_hf_models()` (~110 lines)
- Long functions typically reflect the monolith problem in `app.py`

**Parameters:**
- Maximum ~8 parameters for core domain functions (e.g., `score_model()` takes 7)
- Callback injection via parameter types (e.g., `truncate_plain: _TruncatePlain`, `align_plain: _AlignPlain`)
- Keyword-only arguments with `*` separator used in markup functions

**Return Values:**
- Providers: `tuple[list[dict], list[str]]` — results + errors
- Cache: `dict | None` — None signals cache miss
- Validation: `tuple[bool, str | None]` — valid flag + optional error message

## Module Design

**Exports:**
- Most modules expose all functions publicly (no `__all__`)
- `__init__.py` files re-export key functions where needed
- TypedDicts and dataclasses defined in dedicated files (`core/models.py`, `core/scoring.py`)

**Barrel Files:**
- `providers/__init__.py` — Re-exports `BaseProvider`, provider registry functions, `get_all_provider_classes()`, `get_provider_filter_labels()`
- Other `__init__.py` files are empty or contain minimal re-exports

## Test Conventions

**Test Framework:**
- pytest with class-based test organization
- Classes: `Test*` prefix (e.g., `TestFindGpuBandwidth`, `TestComputeQualityScore`)
- Methods: `test_*` prefix (e.g., `test_known_nvidia_gpu`, `test_perfect_fit_scores_high`)
- Descriptive test names that document the behavior being tested

**Test Pattern:**
```python
class TestComputeQualityScore:
    def test_large_model_scores_higher(self):
        s70b = compute_quality_score(params="70B", quant="Q4_K_M")
        s7b = compute_quality_score(params="7B", quant="Q4_K_M")
        assert s70b > s7b
```

**Fixture pattern:** Minimal — most tests construct inputs inline. `conftest.py` only handles sys.path and `--run-live` flag.

## Markup/Formatting Convention

**Results table cells follow a consistent pattern:**
1. Truncate to available width
2. Align (left/right/center)
3. Apply color based on semantic meaning
4. Return Rich-markup string

```python
def fit_cell_markup(fit_text, *, width, truncate_plain, align_plain) -> str:
    plain = truncate_plain(fit_text, 20)
    plain = align_plain(plain, width, "left")
    lowered = plain.lower()
    if "perfect" in lowered or lowered == "fit":
        return f"[bold #4fe08a]{plain}[/bold #4fe08a]"
    ...
```

---

*Convention analysis: 2026-06-09*
