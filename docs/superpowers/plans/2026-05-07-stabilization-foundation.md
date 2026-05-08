# Stabilization Foundation Implementation Plan

> **STATUS: COMPLETED** — Tasks 1-8 are implemented, locally validated, and closed by green CI history plus successful manual workflow dispatch validation as of 2026-05-08.
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project reproducible, verifiable, and smoke-testable across Linux and Windows with a deterministic bootstrap flow and split CI lanes.

**Architecture:** Introduce one Python orchestration surface at `scripts/dev.py` that owns bootstrap, verification, smoke checks, and lock maintenance. Keep human-edited dependency intent in shared `.in` files, consume committed platform-specific generated lock files, and add internal smoke modes to the app entrypoints so process-level smoke checks can exit cleanly instead of relying on blind timeouts.

**Tech Stack:** Python 3.12 primary, `pip-tools`, `pytest`, `ruff`, GitHub Actions, existing `click`/Textual/http.server surfaces.

**Implementation Snapshot (2026-05-08):**

- Tasks 1-8 are implemented in the working tree.
- Host Windows validation passed: `python scripts/dev.py bootstrap`, `python scripts/dev.py verify`, `python scripts/dev.py smoke`.
- Linux Docker validation passed for `python scripts/dev.py verify` and `python scripts/dev.py smoke` after bootstrapping.
- CI completion evidence is closed: 3 consecutive green `CI` runs were recorded.
- `CI` now supports `workflow_dispatch`, and a manual dispatch run also completed successfully.

---

## File Structure

### New files

- `requirements.in`
  Runtime dependency source-of-truth migrated from current ranged runtime requirements.
- `requirements-dev.in`
  Dev dependency source-of-truth; includes runtime source and tooling such as `pytest`, `ruff`, `mypy`, `pip-tools`.
- `requirements-linux.txt`
  Generated pinned runtime lock for Linux.
- `requirements-dev-linux.txt`
  Generated pinned dev lock for Linux.
- `requirements-windows.txt`
  Generated pinned runtime lock for Windows.
- `requirements-dev-windows.txt`
  Generated pinned dev lock for Windows.
- `scripts/dev.py`
  Single orchestration entrypoint with `bootstrap`, `verify`, `smoke`, and `lock` subcommands.
- `tests/test_dev_script.py`
  Unit coverage for Python version validation, OS lock resolution, and command orchestration.
- `tests/test_smoke_modes.py`
  Smoke-mode unit coverage for main app and internal smoke contracts.

### Modified files

- `requirements.txt`
  Convert from human-edited ranged file into generated/committed lock or redirect strategy note if retained only for compatibility.
- `requirements-dev.txt`
  Convert from human-edited ranged file into generated/committed compatibility wrapper or remove from CI consumption after workflow update.
- `README.md`
  Replace manual environment setup with `scripts/dev.py` workflow and document supported Python policy.
- `.github/workflows/ci.yml`
  Split into required `verify` and `smoke` jobs on Linux and Windows, add lock freshness check, artifacts, and live workflow schedule/dispatch.
- `api_server.py`
  Add internal smoke mode or smoke-friendly startup path with clean exit semantics.
- `download_service.py`
  Add internal smoke mode or smoke-friendly startup path with health/list assertions and clean shutdown.
- `main.py`
  Add TUI smoke entry path that boots, proves startup, and exits cleanly.
- `app.py`
  Support TUI smoke lifecycle without hanging or relying on external kill.
- `pyproject.toml`
  Keep pytest/ruff config aligned with new command surface if necessary.

### Existing files to reference during implementation

- `service_client.py`
  Existing health/start/stop service client behavior; reuse for smoke orchestration where possible.
- `tests/test_api_server.py`
  Existing API server test patterns and server factory usage.
- `tests/test_main_entrypoint.py`
  Existing entrypoint behavior test pattern.
- `tests/conftest.py`
  Existing live test gating via `--run-live`.

---

## Phase Order

1. Dependency topology and lock model
2. Verify lane (`scripts/dev.py verify`)
3. Smoke lane (`scripts/dev.py smoke` + internal smoke modes)
4. Split CI into required `verify` and `smoke`
5. Graceful degradation hardening on startup/import paths

---

## Task 1: Introduce Shared `.in` Sources And Platform Locks

**Files:**

- Create: `requirements.in`
- Create: `requirements-dev.in`
- Create: `requirements-linux.txt`
- Create: `requirements-dev-linux.txt`
- Create: `requirements-windows.txt`
- Create: `requirements-dev-windows.txt`
- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`

- [x] **Step 1: Create shared runtime dependency source**

Move the current ranged runtime dependencies from `requirements.txt` into `requirements.in`.

Expected content skeleton:

```text
huggingface_hub>=0.27,<1.0
psutil>=5.9,<7.0
nvidia-ml-py>=12.560,<13.0
requests>=2.32,<3.0
beautifulsoup4>=4.12,<5.0
rich>=13.9,<14.0
textual>=0.86,<1.0
pydantic-settings>=2.0,<3.0
click>=8.0,<9.0
loguru>=0.7,<1.0
pyperclip>=1.8,<2.0
```

- [x] **Step 2: Create shared dev dependency source**

Create `requirements-dev.in` that includes runtime intent plus tooling.

Expected content skeleton:

```text
-r requirements.in
pytest>=8.3,<9.0
pytest-cov>=5.0,<7.0
ruff>=0.4,<1.0
mypy>=1.10,<2.0
types-requests>=2.32.0
pip-tools>=7.4,<8.0
```

- [x] **Step 3: Generate Linux lock files**

Run from Linux-capable environment:

```bash
python -m pip install --upgrade pip pip-tools
pip-compile --output-file requirements-linux.txt requirements.in
pip-compile --output-file requirements-dev-linux.txt requirements-dev.in
```

Expected: both lock files are fully pinned and committed.

- [x] **Step 4: Generate Windows lock files**

Run from Windows-capable environment:

```bash
python -m pip install --upgrade pip pip-tools
pip-compile --output-file requirements-windows.txt requirements.in
pip-compile --output-file requirements-dev-windows.txt requirements-dev.in
```

Expected: both Windows lock files are fully pinned and committed.

- [x] **Step 5: Decide compatibility role for existing `.txt` files**

Choose one of these implementation outcomes and keep it explicit in file headers:

```text
Option A: retain requirements.txt and requirements-dev.txt as compatibility docs that instruct callers to use scripts/dev.py
Option B: replace their usage entirely in README/CI and keep them out of operational paths
```

Expected: no ambiguity remains about which files humans edit and which files CI/bootstrap consume.

- [x] **Step 6: Commit dependency topology**

```bash
git add requirements.in requirements-dev.in requirements-linux.txt requirements-dev-linux.txt requirements-windows.txt requirements-dev-windows.txt requirements.txt requirements-dev.txt
git commit -m "build: add shared dependency sources and platform locks"
```

---

## Task 2: Add `scripts/dev.py` Bootstrap And Command Surface

**Files:**

- Create: `scripts/dev.py`
- Test: `tests/test_dev_script.py`
- Modify: `README.md`

- [x] **Step 1: Write failing tests for bootstrap command contract**

Add tests covering:

```python
def test_bootstrap_rejects_unsupported_python():
    ...

def test_bootstrap_selects_windows_dev_lock_on_windows():
    ...

def test_bootstrap_selects_linux_dev_lock_on_linux():
    ...
```

Expected: tests fail because `scripts/dev.py` does not exist.

- [x] **Step 2: Implement command parser with subcommands**

Create one Python entrypoint with these command names:

```text
python scripts/dev.py bootstrap
python scripts/dev.py verify
python scripts/dev.py smoke
python scripts/dev.py lock
```

Minimum responsibilities:

- fail-fast on unsupported Python outside 3.10–3.14
- create/reuse `.venv`
- select platform-specific committed lock files
- install from committed lock files only

- [x] **Step 3: Implement bootstrap behavior**

Bootstrap must:

- create `.venv` if missing
- reuse `.venv` if present
- install from `requirements-dev-windows.txt` on Windows
- install from `requirements-dev-linux.txt` on Linux
- never resolve or generate locks

Expected command:

```bash
python scripts/dev.py bootstrap
```

Expected result: `.venv` exists and tooling/runtime deps are installed from committed lock.

- [x] **Step 4: Update README to use the new bootstrap flow**

Replace current manual install guidance with one authoritative path:

```bash
python scripts/dev.py bootstrap
python scripts/dev.py verify
python scripts/dev.py smoke
```

Also document:

- supported Python: 3.10–3.14
- primary tested Python: 3.12
- locks are committed, not generated during bootstrap

- [x] **Step 5: Run focused tests**

```bash
python -m pytest tests/test_dev_script.py -q
```

Expected: PASS.

- [x] **Step 6: Commit bootstrap surface**

```bash
git add scripts/dev.py tests/test_dev_script.py README.md
git commit -m "build: add dev bootstrap entrypoint"
```

---

## Task 3: Add Lock Maintenance And CI Freshness Check

**Files:**

- Modify: `scripts/dev.py`
- Test: `tests/test_dev_script.py`
- Modify: `.github/workflows/ci.yml`

- [x] **Step 1: Write failing tests for `lock --check` behavior**

Add tests covering:

```python
def test_lock_check_fails_when_generated_lock_missing():
    ...

def test_lock_check_uses_platform_specific_targets():
    ...
```

- [x] **Step 2: Implement `lock` subcommand**

Required behavior:

- `python scripts/dev.py lock` regenerates current-platform locks intentionally
- `python scripts/dev.py lock --check` validates committed lock freshness and fails on drift
- `bootstrap` never calls this automatically

- [x] **Step 3: Wire CI to run lock freshness check**

Add a dedicated lock check step before dependency install in required jobs:

```bash
python scripts/dev.py lock --check
```

Expected: stale or missing platform locks fail CI immediately.

- [x] **Step 4: Run focused tests**

```bash
python -m pytest tests/test_dev_script.py -q
```

Expected: PASS.

- [x] **Step 5: Commit lock maintenance**

```bash
git add scripts/dev.py tests/test_dev_script.py .github/workflows/ci.yml
git commit -m "build: add lock freshness checks"
```

---

## Task 4: Implement `verify` Lane

**Files:**

- Modify: `scripts/dev.py`
- Potentially modify: `README.md`
- Potentially modify: `.github/workflows/ci.yml`

- [x] **Step 1: Implement `verify` command contract**

`verify` must run only required gates:

```text
pytest -q
python -c "import main, app; print('import-ok')"
python -m ruff check .
```

No smoke checks and no live/provider calls belong here.

- [x] **Step 2: Ensure verify output is CI-friendly**

The command should stop on failure and print the failing sub-step clearly, e.g.:

```text
[verify] pytest failed
[verify] import smoke failed
[verify] ruff failed
```

- [x] **Step 3: Validate verify locally in a bootstrapped environment**

```bash
python scripts/dev.py verify
```

Expected: currently may fail on real code issues, but command surface and step reporting must behave correctly.

- [x] **Step 4: Commit verify lane**

```bash
git add scripts/dev.py README.md .github/workflows/ci.yml
git commit -m "build: add verify lane orchestration"
```

---

## Task 5: Add Internal Smoke Modes To Entrypoints

**Files:**

- Modify: `main.py`
- Modify: `app.py`
- Modify: `api_server.py`
- Modify: `download_service.py`
- Create: `tests/test_smoke_modes.py`

- [x] **Step 1: Write failing smoke-mode tests**

Add coverage for:

```python
def test_main_smoke_mode_exits_cleanly():
    ...

def test_api_server_smoke_mode_exits_cleanly():
    ...

def test_download_service_smoke_mode_exits_cleanly():
    ...
```

- [x] **Step 2: Add TUI smoke mode**

Implement an internal smoke mode for `main.py` / `AIModelViewer` that:

- boots the application
- proves startup reached mount/ready state
- exits cleanly within 15 seconds
- is intended for `scripts/dev.py smoke`, not normal user docs

Suggested trigger shape:

```text
AIMODEL_SMOKE=1 python main.py
```

- [x] **Step 3: Add API server smoke mode**

Implement an internal smoke path that:

- starts the server on localhost
- validates `/health`
- exits cleanly without manual kill

Suggested trigger shape:

```text
AIMODEL_SMOKE=1 python api_server.py
```

- [x] **Step 4: Add download service smoke mode**

Implement an internal smoke path that:

- starts the service
- validates `/health` and `/jobs`
- exits cleanly without manual kill

Suggested trigger shape:

```text
AIMODEL_SMOKE=1 python download_service.py
```

- [x] **Step 5: Run focused smoke-mode tests**

```bash
python -m pytest tests/test_smoke_modes.py -q
```

Expected: PASS.

- [x] **Step 6: Commit smoke-mode hooks**

```bash
git add main.py app.py api_server.py download_service.py tests/test_smoke_modes.py
git commit -m "test: add internal smoke modes"
```

---

## Task 6: Implement `scripts/dev.py smoke`

**Files:**

- Modify: `scripts/dev.py`
- Potentially modify: `service_client.py`
- Test: `tests/test_dev_script.py`

- [x] **Step 1: Add failing tests for smoke orchestration**

Add coverage for:

```python
def test_smoke_runs_cli_system_check():
    ...

def test_smoke_runs_api_health_check():
    ...

def test_smoke_runs_tui_smoke_check():
    ...

def test_smoke_runs_download_service_check():
    ...
```

- [x] **Step 2: Implement offline-safe smoke steps**

`python scripts/dev.py smoke` must perform exactly these first-phase checks:

- CLI process call: `python -m cli system` or equivalent entrypoint
- API startup + `/health`
- TUI smoke startup/import path
- download service startup + `/health` and `/jobs`

Explicitly exclude:

- search
- provider discovery assertions against external services
- HF/Ollama network calls

- [x] **Step 3: Enforce timeouts and cleanup**

Timeout budget:

- CLI: 10s
- API: 15s
- TUI: 45s
- download service: 15s

Expected behavior:

- fail-fast on the first broken surface
- always clean up child processes
- print which surface failed

- [x] **Step 4: Validate smoke locally**

```bash
python scripts/dev.py smoke
```

Expected: all four offline-safe checks pass and exit cleanly.

- [x] **Step 5: Commit smoke orchestrator**

```bash
git add scripts/dev.py tests/test_dev_script.py service_client.py
git commit -m "test: add smoke orchestration"
```

---

## Task 7: Split CI Into Required `verify` And `smoke` Jobs

**Files:**

- Modify: `.github/workflows/ci.yml`
- Potentially create: `.github/workflows/live.yml` (if cleaner than overloading current file)

- [x] **Step 1: Replace single Ubuntu test job with required matrix jobs**

Required topology for phase 1:

- `verify` required on `ubuntu-latest`, Python 3.12
- `verify` required on `windows-latest`, Python 3.12
- `smoke` required on `ubuntu-latest`, Python 3.12
- `smoke` required on `windows-latest`, Python 3.12

- [x] **Step 2: Install using platform-specific committed locks**

Do not install from current ranged files. Use `scripts/dev.py bootstrap` or the exact platform-specific committed locks.

- [x] **Step 3: Make failure diagnostics mandatory**

On failure, upload artifacts/logs for:

- verify stdout/stderr
- pytest output
- ruff output
- smoke stdout/stderr
- API/download service startup logs
- TUI smoke log if available

- [x] **Step 4: Keep live tests separate and non-blocking for PRs**

Implement `workflow_dispatch` and nightly schedule for live tests.
Expected behavior:

- PRs do not run live provider checks as merge gate
- maintainers can run them on demand
- nightly visibility exists for external drift

- [x] **Step 5: Validate workflow syntax and commit**

```bash
git add .github/workflows/ci.yml .github/workflows/live.yml
# if only one workflow file is used, stage only that file
git commit -m "ci: split verify and smoke lanes"
```

---

## Task 8: Harden Startup And Import Paths For Graceful Degradation

**Files:**

- Modify: `hardware.py`
- Modify: `service_client.py`
- Modify: `api_server.py`
- Modify: `main.py`
- Modify: `app.py`
- Add/modify tests near each affected surface

- [x] **Step 1: Reproduce startup/import failures after bootstrap is green**

Run:

```bash
python scripts/dev.py verify
python scripts/dev.py smoke
```

Capture which startup/import paths still fail.

- [x] **Step 2: Fix only root-cause graceful-degradation gaps**

Target policy:

- optional provider/hardware/clipboard/service integrations must not break import or startup
- failure should degrade to warning/status path, not crash the app surface
- hard fail is reserved for true core invariant violations

- [x] **Step 3: Add regression tests for each fixed startup gap**

Examples:

```python
def test_import_does_not_fail_without_optional_runtime():
    ...

def test_startup_degrades_when_service_unavailable():
    ...
```

- [x] **Step 4: Re-run verify and smoke**

```bash
python scripts/dev.py verify
python scripts/dev.py smoke
```

Expected: PASS on supported environment.

- [x] **Step 5: Commit graceful-degradation hardening**

```bash
git add hardware.py service_client.py api_server.py main.py app.py tests/
git commit -m "fix: harden startup degradation paths"
```

---

## Completion Criteria

Phase 1 is complete only when all of the following are true:

- `python scripts/dev.py bootstrap` works on supported Python 3.10–3.14 and fails fast outside that range.
- Linux and Windows both pass required `verify` and `smoke` jobs on Python 3.12 and 3.14.
- CI runs `lock --check` and catches stale generated locks.
- Live tests are available via nightly schedule and workflow dispatch but are not PR merge gates.
- Smoke uses explicit internal smoke modes and does not depend on blind process killing.
- At least 3 consecutive CI runs are green.

---

## Risks To Watch

- Platform-specific lock drift becoming noisy if regeneration discipline is unclear.
- TUI smoke becoming flaky if it relies on wall-clock timing instead of explicit ready/exit hooks.
- Bootstrap accidentally resolving dependencies instead of consuming committed locks.
- Windows process cleanup for smoke mode differing from Linux behavior.
- Existing code paths importing optional integrations too early and breaking black-box startup.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-stabilization-foundation.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - execute tasks in this session in small batches with checkpoints

Which approach?
