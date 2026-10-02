# Responsive Provider Discovery

## Runtime Behavior

The runtime TUI mounts with the canonical Ollama and Hugging Face options. It discovers optional providers in a Textual background worker after mount and refreshes every 30 seconds. Overlapping timer requests reuse the active probe. Options are applied on the UI thread while retaining the current selection, including a selected optional provider that subsequently becomes unavailable. Programmatic option updates do not dispatch a new search. Search failures continue to use the existing provider diagnostics.

The hardware monitor is initialized lazily by background workers, once per viewer. Runtime startup renders cached hardware immediately, or shows a detection placeholder on a cold start. Ollama process status is checked in the metrics worker, not during cached-header rendering. See [hardware startup](hardware-startup.md) for the shared initialization, failure/retry behavior and remaining startup-budget limits.

REST `/api/v1/providers` owns an availability cache per server, separate from search/model caches. The first request schedules a daemon probe and immediately returns canonical provider descriptors and additive metadata:

```json
{
  "discovery": {
    "status": "pending",
    "refreshing": true,
    "stale": true,
    "error": null
  }
}
```

`status` describes the last probe outcome: `pending` before a completed probe, `ready` after success, or `error` after a failed refresh. `refreshing` independently reports an active background probe. `stale` means no successful snapshot exists or the last successful snapshot is at least 30 seconds old. Thus a failed refresh can return `error` and `stale: true` with retained availability values.

Before the first successful probe, local-runtime availability defaults to false and Hugging Face defaults to true as an integrated remote source. Those defaults are provisional, not connectivity evidence; clients must inspect discovery metadata instead of treating pending booleans as a completed availability assessment. Provider capability metadata remains authoritative for operations independently of runtime reachability.

At most one refresh runs per server. A completed success or failure suppresses another probe for 30 seconds; failure preserves last-known values and returns only the exception class name, never exception text. A stale request schedules refresh on demand and receives retained data immediately. Existing registry detection functions still perform synchronous complete probes for explicit callers.

## Regression and Timing Evidence

TDD first reproduced a timed-out REST request behind a blocked probe and a runtime viewer probing on the UI thread. Further failing tests pinned expired refresh, parallel-request coalescing, failure/retry recovery, and unintended search dispatch during selector option updates. Controlled Events and an injected clock keep these checks independent of external services.

On the local Windows machine, before the fix one normal provider-list request took 33.294 seconds. With the same absent-runtime conditions, 20 requests after the fix returned a first response in 0.0551 seconds, p95 in 0.0249 seconds and maximum in 0.0551 seconds while the actual probe remained pending. These measurements demonstrate response-path separation; they are not upstream-detection completion times or cross-platform latency guarantees. Full hardware/startup benchmarking remains separate work.

Focused checks:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_nonblocking_provider_discovery.py tests/test_provider_discovery_snapshot.py
```
