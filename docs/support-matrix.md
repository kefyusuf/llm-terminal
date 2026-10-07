# Operation-specific support evidence

Evidence reviewed on 2026-10-07. Installation, startup, acquisition and inference
are different operations; success in one does not establish the others.

| Environment | Verified operation | Evidence | Unverified operation |
|---|---|---|---|
| Native Windows, Ollama 0.35.0, tinyllama:1.1b | Isolated managed acquisition, partial cancel, server/service reopen, retry, manifest/blob hashes; forced-CPU repeated inference | [Named-model trial](ollama-live-acceptance.md) | CUDA generation failed at 30 seconds; GPU calibration, other models/platforms and power loss remain unqualified |
| Native Windows, Python 3.12.14, HF SDK 0.36.2 | CLI/TUI/API/service smoke, pinned 19 MB HF acquisition, cancellation/retry, controlled and forced service stop with child exit, selected-file removal, isolated previous-candidate worker/file restore | [HF trial](hf-recovery-acceptance.md), [service trial](managed-download-removal.md), [forced-stop trial](evidence/hf-service-crash-windows-2026-10-04.json), [candidate restore](qualified-candidate-restore.md) | Host/power-loss recovery, other inference/backend combinations and broad calibration |
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

The refreshed 2026-10-05 development-host probe found the installed Ollama client
0.35.0. The default loopback API was unavailable and the local model store had
zero manifests. This supersedes the earlier CLI-unavailable observation. Official
release metadata for v0.35.1 listed a 1,471,094,402-byte Windows amd64 portable
archive; checking that metadata did not install the runtime. A dedicated running
server, isolated model store and declared model artifact/time budget are still
needed for the Ollama acceptance gate. A subsequent [owned-server preflight](evidence/ollama-runtime-preflight-windows-2026-10-05.json)
on source `5906e79` started runtime 0.35.0 on a separate loopback port with a
dedicated store and process-only `OLLAMA_NO_CLOUD=1`. The real `/api/status`
confirmed cloud disabled, `/api/tags` returned zero models, and the owned process
was stopped. Only empty `blobs`/`manifests` directories were created. Default
server unavailability is distinct from this successful dedicated-server startup.
That preflight did not download or execute a model. The later
[2026-10-07 named-model trial](ollama-live-acceptance.md) verified acquisition and
CPU inference; its CUDA calibration failed and remains unqualified. See the
official [local-only configuration](https://docs.ollama.com/faq#how-do-i-disable-ollama-cloud-features).

Stable promotion must not advertise untested inference/backend combinations.
A restricted prerelease can label optional runtimes/platform operations
experimental. The mandatory HF/Ollama acquisition/recovery gate has bounded
named-artifact native Windows evidence and separately recorded hosted HF evidence;
this does not certify all runtime/backend/model combinations. Failed CUDA
generation and the remaining release/pilot gates must stay explicit.
