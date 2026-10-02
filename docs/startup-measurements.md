# Local Startup Measurements

## Reproduce

From a bootstrapped checkout, run the opt-in development harness:

```powershell
& .\.venv\Scripts\python.exe scripts/benchmark_startup.py --runs 20 --machine my-test-machine --output .venv/startup-benchmark.json
```

The command exercises the real runtime viewer in 120x40 headless Textual sessions. Each repetition uses a fresh Python process and separate empty cache/download/model paths under `.venv/startup-benchmark-runs/`. It does not submit model searches, queue downloads, change trust settings, or stop shared services. The normal download-service readiness path remains active and may reuse/start the standard local service; its state is not reset between runs.

The parent stops its own measurement child after receiving a complete sample. Optional probes can still be waiting on upstream timeouts at that point. This avoids charging probe cleanup to UI/request response latency; it does not measure successful probe completion or prove worker shutdown/recovery. Probe absence/timeout/recovery and interaction behavior have separate regression coverage.

## Metric Definitions

| Metric | Start | End | Provisional p95 target |
| --- | --- | --- | --- |
| `startup_ms` | Parent starts the fresh Python process | Parent receives the flushed marker after the runtime headless session is mounted | <= 2,000 ms |
| `input_ms` | Pilot begins injecting one `x` key into the focused input | The message hook observes the matching `Input.Changed` event | <= 100 ms |
| `providers_ms` | Begin loopback HTTP request to a fresh REST server | Provider-list response body is decoded | <= 2,000 ms |

Startup includes interpreter launch, script/application imports, viewer construction, mount and marker transport. It is a conservative headless usability measurement, not terminal paint timing. Input excludes the Pilot's later idle/render wait; it measures dispatch-to-input-change, not screen paint. REST timing excludes server construction and hardware initialization. The REST server reuses the viewer's real completed monitor and returns its first availability snapshot.

Each sample records whether input was handled while TUI provider detection was still active and the REST discovery metadata. Hugging Face's provisional availability is an integrated source, not connectivity evidence. CPU/GPU names, OS/Python, an explicit machine label and the tested source revision identify the context without exposing tokens, headers, hostname or generated storage paths.

The report uses nearest-rank p95 (`ceil(0.95 * n)`) and retains maximums and all samples. Twenty successful repetitions are required. Missing, failed, negative/nonfinite measurements or missed p95 targets cannot produce a passing budget. `--runs 1` can validate the harness but returns a nonpassing result. Exit `0` means this local measurement budget passed; exit `2` means it did not. Neither authorizes a release.

Do not run resource-intensive tests or other benchmarks alongside the measurement if comparing baselines. Optional runtime presence, machine load, filesystem cache and download-service reuse affect results and must be recorded. Cold application state does not mean cold OS filesystem cache. Host-specific p95 targets are not global product SLAs.

## Evidence Scope

Windows absence-case results will be recorded with a named-machine baseline and raw sanitized JSON. This does not validate an available optional-runtime scenario, Linux/macOS terminal paint budgets, or the full release roadmap. Preserve previous before/after request evidence in `provider-discovery.md` and the single-session hardware probe in `hardware-startup.md`.
