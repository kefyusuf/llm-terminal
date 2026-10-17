# In-process download refactor validation

Date: 2026-10-17. Worktree: `E:\projects\llm-terminal\.worktrees\inproc-downloads`.
Branch: `refactor/inproc-downloads`. Baseline: `a16fba3`.

The TUI owns one download manager with bounded worker threads, per-job cancellation,
thread-safe progress snapshots, and SQLite history. Transfers use in-process HTTP.
HF uses Hub metadata, immutable URLs and the SDK-compatible partial-file layout.
Ollama uses streaming `POST /api/pull`. Disk, path, identity, byte-count and digest
checks remain enforced. Progress persistence is throttled while snapshots stay live.
SQLite operation connections now close after commit or rollback, including on Windows.

## Verification

All commands ran from the worktree root using `.venv\Scripts\python.exe`.
TEMP, TMP and the default cache/download/model paths were process-scoped to
`.test-tmp` inside this worktree because the sandbox cannot write user data paths.
No provider model downloads, package installations or Docker resources were started.

- Final full-suite result: **878 passed, 1 skipped** (879 collected); exit code 0.
- Ruff and smoke: exit code 0.

Verbatim output tails (pytest, Ruff, smoke respectively):

```text
...............                                                          [100%]
All checks passed!
[smoke] api server ok
download-manager-smoke-ok
```

- Full pytest command: `python -m pytest -q -p no:cacheprovider`.
- Ruff command: `python -m ruff check .`.
- Smoke command: `python scripts/dev.py smoke`.
- `git diff --check` passed.
- The initial manager test failed with the expected missing-module error before implementation.
- Focused tests cover concurrent same-target dedupe, progress snapshots, cancellation,
  worker bounds, queued cancel/delete/requeue, shutdown, restart recovery, blocked
  plans, failures, SDK partial resume, stream framing, file identity and hash cancellation.
- Independent review found no remaining concrete must-fix after the fixes.

## Behavior and limits

- Intentional change: app exit stops downloads. Interrupted jobs remain visible and
  resumable on next submission; startup does not automatically resume them.
- Live percentage, speed and ETA, blocked explanations, and duplicate prevention remain.
  Transfer speed/ETA are calculated from received byte counts; Ollama metrics remain per layer.
- HTTP download-service endpoints, detached lifetime, token/port settings, protocol
  upgrades and the old IPC diagnostics are removed.
- Existing SQLite columns/rows remain compatible; abandoned queued/running/canceling jobs
  become `interrupted`. HF SDK partial files remain compatible.
- Existing Ollama model-data removal still uses `ollama rm`; transfers do not spawn children.
- Real HF/Ollama transfers, cross-platform runtime behavior and built-package consumption
  were not executed in this run. Installed-package and recovery harnesses were adapted
  and their deterministic tests passed.
- HF partial-path integration uses `huggingface_hub._local_folder` helpers. This private SDK
  interface should be rechecked on a dependency upgrade.
- Cancellation is cooperative and can wait for an active network read/metadata timeout.
- Initial untracked `graft/` was preserved. Owned `.pytest-integration*` and
  `.test-tmp*` test directories remain untracked after cleanup was denied by policy.
  No Docker resources were created, removed or retained.

## Commit blocker

No commits or pushes were made. The staging attempt failed verbatim:

```text
fatal: Unable to create 'E:/projects/llm-terminal/.git/worktrees/inproc-downloads/index.lock': Permission denied
```

The shared Git metadata directory is outside the writable roots and approval escalation
is disabled in this session. No alternate index, ref or object-store workaround was used.
The source changes remain in the requested worktree.

Suggested logical commits when Git metadata writes are available:

1. `refactor: run downloads inside the application` (code, scripts and tests).
2. `docs: describe application-owned download lifecycle` (README and docs).

Both commit messages must end with a blank line followed by:
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File and line changes

Counts are additions/deletions relative to the baseline; new files are counted in full.
The compatibility test file is listed as delete/add because staging is unavailable.

| File | Status | Added | Deleted |
| --- | --- | ---: | ---: |
| `README.md` | Changed | 8 | 13 |
| `app/download_manager.py` | Changed | 52 | 73 |
| `app/startup_viewer.py` | Changed | 13 | 37 |
| `cli.py` | Changed | 0 | 8 |
| `config.py` | Changed | 1 | 5 |
| `core/diagnostics.py` | Changed | 0 | 15 |
| `docs/doctor.md` | Changed | 4 | 4 |
| `docs/installed-package-validation.md` | Changed | 4 | 3 |
| `docs/managed-download-removal.md` | Changed | 35 | 58 |
| `docs/qualified-candidate-restore.md` | Changed | 8 | 0 |
| `docs/structure.md` | Changed | 7 | 4 |
| `downloads/__init__.py` | Changed | 1 | 1 |
| `downloads/api.py` | Deleted | 0 | 265 |
| `downloads/backends.py` | Added | 287 | 0 |
| `downloads/download_lifecycle.py` | Changed | 0 | 32 |
| `downloads/download_manager.py` | Changed | 1 | 1 |
| `downloads/download_service.py` | Deleted | 0 | 317 |
| `downloads/download_status.py` | Changed | 3 | 1 |
| `downloads/hf_downloader.py` | Deleted | 0 | 58 |
| `downloads/manager.py` | Added | 188 | 0 |
| `downloads/parent_lifetime.py` | Deleted | 0 | 23 |
| `downloads/preflight.py` | Changed | 7 | 1 |
| `downloads/progress.py` | Deleted | 0 | 23 |
| `downloads/runner.py` | Deleted | 0 | 518 |
| `downloads/service_client.py` | Deleted | 0 | 282 |
| `downloads/store.py` | Changed | 16 | 51 |
| `pyproject.toml` | Changed | 0 | 3 |
| `scripts/dev.py` | Changed | 7 | 11 |
| `scripts/verify_hf_recovery.py` | Changed | 46 | 73 |
| `scripts/verify_installed_package.py` | Changed | 5 | 63 |
| `tests/test_cache_db.py` | Changed | 1 | 0 |
| `tests/test_dev_script.py` | Changed | 6 | 11 |
| `tests/test_doctor.py` | Changed | 1 | 2 |
| `tests/test_download_admission_feedback.py` | Deleted | 0 | 191 |
| `tests/test_download_admission_response_loss.py` | Deleted | 0 | 108 |
| `tests/test_download_backends.py` | Added | 322 | 0 |
| `tests/test_download_duplicate_submissions.py` | Deleted | 0 | 278 |
| `tests/test_download_idle_retirement.py` | Deleted | 0 | 138 |
| `tests/test_download_lifecycle.py` | Changed | 0 | 20 |
| `tests/test_download_manager.py` | Changed | 68 | 33 |
| `tests/test_download_preflight.py` | Changed | 0 | 87 |
| `tests/test_download_progress.py` | Deleted | 0 | 90 |
| `tests/test_download_queue_capacity.py` | Deleted | 0 | 134 |
| `tests/test_download_service_auth.py` | Deleted | 0 | 173 |
| `tests/test_download_service_debug.py` | Deleted | 0 | 36 |
| `tests/test_download_service_startup.py` | Deleted | 0 | 457 |
| `tests/test_download_service_store.py` | Deleted | 0 | 185 |
| `tests/test_download_service_worker.py` | Deleted | 0 | 209 |
| `tests/test_download_store_compatibility.py` | Added | 185 | 0 |
| `tests/test_download_sync_errors.py` | Changed | 7 | 5 |
| `tests/test_downloads_runner.py` | Deleted | 0 | 92 |
| `tests/test_downloads_runner_execution.py` | Deleted | 0 | 454 |
| `tests/test_downloads_store.py` | Changed | 68 | 39 |
| `tests/test_downloads_streamed_cancel.py` | Deleted | 0 | 75 |
| `tests/test_hf_models_dir.py` | Changed | 0 | 9 |
| `tests/test_hf_parent_lifetime.py` | Deleted | 0 | 84 |
| `tests/test_hf_process_recovery.py` | Deleted | 0 | 87 |
| `tests/test_hf_revision_pinning.py` | Changed | 1 | 21 |
| `tests/test_hf_single_file_download.py` | Changed | 0 | 19 |
| `tests/test_inprocess_download_manager.py` | Added | 163 | 0 |
| `tests/test_installed_package.py` | Changed | 1 | 16 |
| `tests/test_managed_download_removal.py` | Changed | 0 | 197 |
| `tests/test_narrow_search_filter_layout.py` | Changed | 3 | 1 |
| `tests/test_nonblocking_hardware_startup.py` | Changed | 3 | 2 |
| `tests/test_nonblocking_provider_discovery.py` | Changed | 3 | 1 |
| `tests/test_ollama_download_controls.py` | Deleted | 0 | 268 |
| `tests/test_runtime_endpoint_configuration.py` | Changed | 0 | 45 |
| `tests/test_service_client.py` | Deleted | 0 | 23 |
| `tests/test_service_client_lifecycle.py` | Deleted | 0 | 280 |
| `tests/test_smoke_modes.py` | Changed | 0 | 29 |
| `tests/test_streamed_pipe_liveness.py` | Deleted | 0 | 212 |
| `tests/test_tui_live_first_stale_fallback.py` | Changed | 3 | 0 |
| `tests/test_tui_startup_service_worker.py` | Changed | 4 | 11 |
| `tui_app.py` | Changed | 7 | 19 |
| `docs/inprocess-downloads-validation.md` | Added | 166 | 0 |

Totals including this report: 75 files; 1705 added lines, 6084 deleted lines.
