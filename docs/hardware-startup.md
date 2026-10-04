# Responsive Hardware Startup

## Behavior

Viewer construction does not run CPU-name lookup, NVML initialization, or vendor subprocess probes. The first metrics worker initializes `HardwareMonitor`; search and model-detail workers share that monitor through a per-viewer initialization lock. Only successful construction is published, so concurrent workers cannot create duplicate monitors and a failed constructor can be retried.

The runtime entry point (`main.py` -> `app.viewer` -> `app.startup_viewer`) renders cached hardware immediately, with Ollama status marked `checking` until the background process scan completes. A cold start shows `Detecting hardware...`. UI-side search-cache compatibility checks use the latest/cached snapshot or a conservative pending placeholder; they never initiate detection. Submitted searches wait for actual hardware in their background worker before calling providers.

When initial metrics detection fails, the cold header reports that polling will retry. Cached data remains available during refresh failure. Expected `OSError`, `ValueError` and `RuntimeError` failures in search or detail workers produce a retry diagnostic instead of a Textual worker panic. Periodic metrics polling or the next user operation can retry initialization. Unexpected programming errors remain visible.

Smoke-mode TUI startup does not initiate hardware detection. CLI/REST hardware construction and explicit `HardwareMonitor` callers retain their current behavior.

## Verification

The regression suite exercises mounted cold/cached input, a submitted search while detection is blocked, fresh hardware delivery to the provider, a shared monitor across concurrent workers, failure/retry recovery, and search/detail failures that keep the app open. CPU/GPU probes and process scans are replaced only at their external boundary; the Textual lifecycle, selectors, search orchestration and completion behavior run normally.

TDD reproduced UI-thread construction, an incorrect stopped Ollama label before a completed check, duplicate concurrent monitors, missing failure feedback, and Textual worker termination on expected hardware failure before the corresponding fixes.

The mounted-search regression also covers a debounce delay longer than a single pilot pause. It waits for observable search-start and completion state with bounded deadlines and releases the blocked probe before teardown. This replaces the fixed-duration assumption that failed on Windows hosted runners; all behavioral assertions remain in place.

The full suite also exposed `test_ollama_search_returns_models_with_mocked_html` patching the HTTP factory in its defining module instead of the provider's imported binding. Its corrected patch now uses the intended fake HTML without reaching the real registry.

A local Windows cold-cache probe used the real runtime viewer, CPU/GPU detection, provider discovery and download-service lifecycle without requesting model weights. Construction took 0.0104 seconds and entry into the mounted headless Textual session took 0.2824 seconds. Search input accepted text, and the background snapshot identified the Ryzen 5 7500F and RTX 4060 Ti. These are one-machine headless measurements, not terminal paint timings or cross-platform guarantees.

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_nonblocking_hardware_startup.py
```

## Remaining Startup Work

This completes the CPU/GPU construction and cached-header process-scan part of R1. It does not establish a universal first-frame latency guarantee. Runtime cache initialization/cleanup and history rendering still occur during mount, and cross-platform timing budgets require measurement. Optional provider discovery follows its separate background lifecycle. The legacy base viewer also retains its existing synchronous download-service mount path; the runtime entry point uses the asynchronous startup subclass.
