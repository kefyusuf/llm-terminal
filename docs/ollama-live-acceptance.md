# Bounded native Windows Ollama acceptance

On 2026-10-07, the user authorized `tinyllama:1.1b` with a 1,000,000,000-byte
model-file limit and 900-second wall budget. Source
`bc50cf1e6bbf2cf19b40833cbe0edae1b910d9ce` passed main CI 11/11 before the trial.
Runtime 0.35.0 used a project-owned store, a separate loopback endpoint and
process-only cloud disabling, verified through the real `/api/status` API.
No shared runtime, global configuration or package registry was changed.

## Acquisition and recovery

The [acquisition report](evidence/ollama-recovery-windows-2026-10-07.json) records
637,700,138 declared artifact bytes and manifest digest
`2644915ede352ea7bdfaff0bfac0be74c719d5d5202acb63a6fb095b52f394a4`.
The managed service retained an active duplicate, acknowledged cancellation
after CLI-reported 1% progress, then stopped. The owned Ollama server stopped too.
After reopening both against the same isolated store/database, the canceled job
persisted and managed retry completed. The stored manifest matched the registry
manifest; every config/model/template/system/parameter blob matched its declared
size and SHA-256. Peak observed store size was 637,702,198 bytes.

The partial file was preallocated to its final size: its length is not downloaded
bytes. CLI progress is a provider report. Store size is not cumulative transfer,
and no transfer-byte reuse is claimed. This qualifies this named model's
controlled acquisition/reopen path, not arbitrary models or power loss.

## CUDA failure and CPU fallback

The existing bounded calibration tool failed on the default CUDA path: the real
generate request timed out after 30 seconds and the server reported HTTP 500.
Logs detected an RTX 4060 Ti and reported CUDA layer offload; successful model
loading is not successful generation. No calibration result was saved. The
acquisition report intentionally retains `status: failed` for this combined trial
even though its separate acquisition/recovery assertions passed. The cause is
unresolved; no driver/runtime downgrade, increased request budget or scoring
change was attempted.

The [CPU fallback report](evidence/ollama-cpu-inference-windows-2026-10-07.json)
used the same verified files without another download or store copy. Process-only
GPU visibility settings and `num_gpu: 0` forced CPU execution. One excluded warmup
and three measured requests used the fixed hash-explanation prompt, context 512
and at most 32 generated tokens. The runtime reported `size_vram: 0`; median
generation throughput was 81.04589730471737 tokens/s. Prompt-cache counts and
HTTP/runtime timing fields remain separate. No response text was archived.
These are actual CPU samples, not a successful run of the failed CUDA calibration
tool, a GPU prediction validation or a quality benchmark. See official
[GPU selection](https://docs.ollama.com/gpu#gpu-selection).

## Cleanup and remaining gates

### 2026-10-08 bounded CUDA follow-up

The unchanged tool passed twice before implementation, so the original transient
30-second failure does not establish a driver, CUDA graphs or cold-start cause.
A real local HTTP regression independently reproduced combined cold loading plus
generation exceeding the request timeout. The correction separates promptless
preload from generation while preserving the 30-second request and 120-second
worker budgets used here, token limits, identity checks and numerical scoring.

The [GPU report](evidence/ollama-cuda-calibration-windows-2026-10-08.json) binds
the tested tool and scoring hashes. Two fresh owned Ollama 0.35.0 servers reused
the verified model above, with cloud disabled and default GPU selection. Each run
completed a separate preload, one excluded warmup and three 64-token samples at
context 512. Median generation rates were 337.86 and 334.51 tokens/s; the runtime
reported 658,379,898 VRAM bytes. Both owned servers stopped with zero remaining
captured descendants. No model was downloaded, driver changed or timeout raised.
This is bounded native Windows GPU evidence for this model and workload, not a
general CUDA reliability guarantee or proof of the original failure's cause.

The acquisition/CUDA phase lasted 125.062 seconds; the CPU probe lasted 4.312
seconds. Preparation and diagnosis also fit the authorized total 15-minute
window. The first preparation attempt lacked the required model name, was
rejected before downloading, and stopped its owned processes.

After the CUDA phase stopped its direct handles, two captured descendants needed
separate cleanup. Only recorded trial descendants with matching PID and creation
time were terminated; zero remained. The CPU probe unloaded the model and stopped
its owned server and descendants. Downloaded files and bounded reports remain
under ignored `.venv/` for review; no model deletion was requested. No Docker
resources or worktrees were created.

Native Windows Ollama acquisition/recovery and repeated CUDA calibration now have
named-model evidence. The historical failed trial remains failed. Other platforms,
a broader calibration corpus, participant outcomes and authorized release
version/publisher/environment/publication remain open. Keep numerical scoring
unchanged until comparable measured evidence supports a change.
