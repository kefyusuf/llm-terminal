# Operation-specific support evidence

Evidence reviewed on 2026-10-04. Installation, startup, acquisition and inference
are different operations; success in one does not establish the others.

| Environment | Verified operation | Evidence | Unverified operation |
|---|---|---|---|
| Native Windows, Python 3.12.14, HF SDK 0.36.2 | CLI/TUI/API/service smoke, pinned 19 MB HF acquisition, cancellation/retry, controlled and forced service stop with child exit, selected-file removal, isolated previous-candidate worker/file restore | [HF trial](hf-recovery-acceptance.md), [service trial](managed-download-removal.md), [forced-stop trial](evidence/hf-service-crash-windows-2026-10-04.json), [candidate restore](qualified-candidate-restore.md) | Host/power-loss recovery, Ollama acquisition, model inference/calibration |
| Hosted Linux x86_64, Python 3.10–3.14 | Isolated wheel installation and entrypoint/service startup | Package CI reports; Python 3.12 report observed Linux 6.17.0/glibc 2.39 | Native GPU/backend and acquisition/recovery on Python versions other than the separately tested 3.12 lane |
| Hosted Linux/Windows, Python 3.12, HF SDK 0.36.2 | Pinned 19 MB HF acquisition, early/partial cancellation, reopen/retry and final size/digest | [Exact-source CI reports](qualified-candidate-restore.md#hosted-acquisition-evidence) | Hosted forced-service crash, Ollama acquisition and GPU inference |
| Hosted macOS arm64, Python 3.10–3.14 | Isolated wheel installation and entrypoint/service startup | Package CI reports; Python 3.12 report observed macOS 26.6.2 arm64 | MLX model loading, GPU inference and real acquisition/recovery |
| Hosted Linux/Windows/macOS, Python 3.12 | Sdist installation and startup | Three independent consumers of the same source archive | Sdist installation on every other Python minor version |
| WSL, Apple Silicon MLX runtime, Docker Model Runner, LM Studio | Deterministic fixtures/optional detection contracts where tested | Existing tests, not runtime certification | Prepared-host live acceptance; experimental |

The candidate pipeline retains 15 wheel and three sdist consumers. Reports include
resolved dependency versions rather than claiming all lower dependency bounds
were exercised. The build-once [candidate contract](build-once-candidates.md)
records exactly which source and bytes each consumer verifies. Hosted macOS is
now an observed arm64 installation environment, not assumed Intel or MLX proof.

The native Windows smoke detected an AMD Ryzen 5 7500F, 31.4 GB total RAM and
RTX 4060 Ti with 16 GB VRAM/CUDA classification. Detection is not allocation,
throughput, driver compatibility or inference evidence. Available RAM/VRAM is
a changing snapshot, not a stable model capacity promise.

Ollama CLI and localhost API were unavailable on the development host. Official
release metadata for v0.35.1 listed a 1,471,094,402-byte Windows amd64 portable
archive; checking that metadata did not install the runtime. A prepared runtime,
isolated model store and declared artifact/time budget are still needed for the
Ollama acceptance gate. An arm64 package is not a substitute on this x64 host.

Stable promotion must not advertise untested inference/backend combinations.
A restricted prerelease can label optional runtimes/platform operations
experimental, but the mandatory HF/Ollama acquisition/recovery gate remains
explicitly incomplete until its actual prepared-host evidence exists.
