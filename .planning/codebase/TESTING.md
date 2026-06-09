# Testing Patterns

**Analysis Date:** 2026-06-09

## Test Framework

**Runner:**
- pytest
- Config: `pyproject.toml` > `[tool.pytest.ini_options]`
- Test path: `tests/`
- Default args: `-q` (quiet mode)

**Assertion Library:**
- Standard `assert` statements (pytest native)

**Run Commands:**
```bash
pytest                          # Run all tests (quiet mode)
pytest -v                       # Verbose output
pytest -x                       # Stop on first failure
pytest --run-live               # Include live integration tests
pytest tests/test_scoring.py    # Single test file
pytest -k "test_perfect"        # Keyword filter
```

**Coverage:**
```bash
pytest --cov=. --cov-report=term --cov-report=html
coverage report                 # Show coverage with missing lines
```

## Test File Organization

**Location:**
- All tests in `tests/` directory (co-located test dir, not co-located with source)
- 34 test files, 3172 lines total

**Naming:**
- `test_{module_name}.py` — Pattern matches the module being tested (e.g., `test_scoring.py` for `core/scoring.py`, `test_hardware_expanded.py` for `core/hardware.py`)
- `test_{feature}_helpers.py` — Helper-specific tests (e.g., `test_hf_provider_helpers.py`, `test_ollama_provider_helpers.py`)

**Structure:**
```
tests/
├── conftest.py                         # Fixtures, --run-live flag
├── test_scoring.py                     # Scoring engine (303 lines)
├── test_hardware_expanded.py           # Hardware detection (79 lines)
├── test_model_intelligence.py          # MoE + quant (214 lines)
├── test_helpers.py                     # Core utils (53 lines)
├── test_search_cache.py                # In-memory cache (81 lines)
├── test_search_orchestration.py        # Search pipeline (64 lines)
├── test_results_layout.py              # Column layout (83 lines)
├── test_results_presenter.py           # Cell markup (82 lines)
├── test_results_text.py                # Text helpers (34 lines)
├── test_results_view.py                # Filter/sort (95 lines)
├── test_download_manager.py            # Download commands (38 lines)
├── test_download_status.py             # Status helpers (27 lines)
├── test_download_history.py            # History helpers (40 lines)
├── test_download_lifecycle.py          # Lifecycle (118 lines)
├── test_download_service_store.py      # Job store (154 lines)
├── test_download_service_debug.py      # Service debug (39 lines)
├── test_service_client.py              # HTTP client (13 lines)
├── test_providers_extended.py          # Provider integration (180 lines)
├── test_hf_provider_helpers.py         # HF helpers (13 lines)
├── test_ollama_provider_helpers.py     # Ollama helpers (59 lines)
├── test_provider_error_helpers.py      # Error handling (25 lines)
├── test_plan_mode.py                   # Plan mode modal (79 lines)
├── test_scoring_ui.py                  # Scoring UI (205 lines)
├── test_app_compact_mode.py            # Compact mode (85 lines)
├── test_app_modal_polling.py           # Modal polling (72 lines)
├── test_smoke_modes.py                 # Smoke tests (89 lines)
├── test_api_server.py                  # REST API (168 lines)
├── test_main_entrypoint.py             # entry point (14 lines)
├── test_themes.py                      # Themes (58 lines)
├── test_cache_db.py                    # SQLite cache (50 lines)
├── test_live_platforms.py              # Live integration (80 lines)
├── test_release_check_helpers.py       # Release check (20 lines)
├── test_dev_script.py                  # Dev script (424 lines)
└── __init__.py                         # Empty
```

## Test Structure

**Suite Organization:**
```python
# Class-based grouping of related tests
class TestFindGpuBandwidth:
    def test_known_nvidia_gpu(self):
        bw = find_gpu_bandwidth("NVIDIA GeForce RTX 4090")
        assert bw == 1008

    def test_unknown_gpu_returns_none(self):
        bw = find_gpu_bandwidth("Some Random GPU 3000")
        assert bw is None

class TestComputeQualityScore:
    def test_large_model_scores_higher(self):
        s70b = compute_quality_score(params="70B", quant="Q4_K_M")
        s7b = compute_quality_score(params="7B", quant="Q4_K_M")
        assert s70b > s7b
```

**Patterns:**
- Class names: `Test{Feature}` (e.g., `TestFindGpuBandwidth`, `TestScoreModel`, `TestComputeFitScore`)
- Method names: `test_{behavior}` (e.g., `test_known_nvidia_gpu`, `test_cpu_offload_slower`, `test_all_weights_sum_to_one`)
- No fixtures in most tests — inputs constructed inline
- No teardown needed (no persistent state in unit tests)

## Mocking

**Framework:**
- No mocking library detected. Tests rely on:
  - Direct function calls with controlled inputs
  - `--run-live` flag for integration tests that hit real services
  - `conftest.py` skips live tests by default

**Patterns:**
```python
# No mocking — test pure functions with known inputs
def test_perfect_fit_scores_high(self):
    score = compute_fit_score(model_size_gb=12.0, vram_gb=24.0, ram_gb=32.0, mode="GPU")
    assert score >= 80
```

**What to Mock:**
- External HTTP calls (HuggingFace API, Ollama registry)
- Hardware detection (GPU availability, VRAM size)
- Subprocess calls (ollama rm, rocm-smi)

**What NOT to Mock:**
- Scoring logic (tested directly with known inputs)
- Layout/formatting functions (tested with sample data)
- Cache logic (tested with controlled state)
- Model intelligence (tested with model name strings)

## Fixtures and Factories

**Test Data:**
```python
# Inline test data — no factory functions
specs = {
    "vram_total": 24.0,
    "vram_free": 20.0,
    "ram_total": 32.0,
    "ram_free": 28.0,
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "has_gpu": True,
}
```

**Location:**
- No centralized test fixtures or factories
- All test data constructed inline in test methods
- `conftest.py` only provides pytest configuration (sys.path, --run-live, markers)

## Coverage

**Requirements:**
```toml
[tool.coverage.report]
fail_under = 40  # CI gate — 40% minimum
```

**View Coverage:**
```bash
coverage run -m pytest && coverage report
coverage html    # Open htmlcov/index.html
```

**Known gaps:** The `fail_under = 40` target is low, suggesting significant untested areas. The monolith `app.py` (2466 lines) has very limited test coverage — only 3 test files target the TUI (`test_app_compact_mode.py`, `test_app_modal_polling.py`, `test_smoke_modes.py`).

## Test Types

**Unit Tests (dominant):**
- Pure function tests for scoring, hardware detection, model intelligence, utilities
- No external dependencies, fast execution
- Examples: `test_scoring.py`, `test_hardware_expanded.py`, `test_helpers.py`

**Service/Store Tests:**
- Tests for `DownloadStore` SQLite operations (`test_download_service_store.py`)
- Tests for download lifecycle (`test_download_lifecycle.py`)
- Uses `DownloadStore.__init__` with temp/memory DB

**Smoke Tests:**
- `test_smoke_modes.py` — Smoke-mode TUI test via `app.run_test()`
- `test_app_compact_mode.py` — Compact-mode UI behavior
- `test_app_modal_polling.py` — Modal poll pause behavior

**Live Integration Tests:**
- `test_live_platforms.py` — Tests that hit real HuggingFace/Ollama APIs
- Gated behind `--run-live` flag
- Skipped by default

**UI Tests:**
- `test_results_layout.py`, `test_results_presenter.py`, `test_results_text.py`, `test_results_view.py` — Test formatting/computation (no actual widget rendering)

**E2E Tests:**
- Not used. Smoke tests (`test_smoke_modes.py`) are the closest equivalent.

## Common Patterns

**Async Testing:**
```python
# Textual's async test pilot — used in smoke tests
async def test_smoke_mode():
    app = AIModelViewer()
    async with app.run_test() as pilot:
        await pilot.pause(1.5)
    assert app.return_code in (None, 0)
```

**Error Testing:**
```python
# Testing error conditions with known edge cases
def test_no_fit_returns_zero(self):
    tok_s = estimate_tok_per_s(
        model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 4090", mode="No Fit"
    )
    assert tok_s == 0.0

def test_empty_string(self):
    bw = find_gpu_bandwidth("")
    assert bw is None
```

**State Machine Testing:**
```python
# Download lifecycle state transitions
class TestDownloadLifecycle:
    def test_cancel_before_delete_required_for_active(self):
        assert should_cancel_before_delete(delete_data=True, state="downloading") is True
        assert should_cancel_before_delete(delete_data=True, state="completed") is False
```

**Hardware-Agnostic Testing:**
```python
class TestHardwareMonitorSpecs:
    def test_specs_has_backend_field(self):
        monitor = HardwareMonitor()
        specs = monitor.get_specs()
        assert "backend" in specs
        assert specs["backend"] in ("cuda", "rocm", "metal", "sycl", "cpu")
```

---

*Testing analysis: 2026-06-09*
