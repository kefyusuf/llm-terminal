# Bounded Hugging Face recovery acceptance

## Installed network interruption: 2026-10-14

The [Windows network interruption report](evidence/hf-network-recovery-windows-2026-10-14.json)
uses the same pinned sample and installed PR #156 wheel as the current-candidate
trial below. Main `6e6832b6631f6aa01e3c5817df910a31ad1c5db5` adds only PR #157
documentation to the qualified runtime. The installed receipt, five module
hashes and final file size/SHA-256 were independently checked again.

A task-owned loopback HTTPS CONNECT relay routes only the trial service and its
HF children. TLS remains end-to-end and certificate verification stays enabled;
no host network setting changes. The relay shuts down active tunnel sockets when
observed encrypted server bytes reach 12 MiB and rejects new tunnels until
restored. The SDK had written a 10 MiB `.incomplete` file when this gate was seen.

The worker exited nonzero, the job persisted `failed` with a diagnostic, the
incomplete artifact was not marked completed, and the service remained healthy.
Service restart retained the failed status and exact download plan. Restoring
the relay and explicitly retrying reused the original job and completed the
19,077,344-byte artifact with its expected digest. All owned services, children
and the relay stopped; new and previous trial files were retained. Application
code was unchanged because this acceptance found no application defect.

This proves handling of an established transfer connection being closed. No
subsequent tunnel request occurred while blocked, so SDK automatic reconnection
is unassessed. DNS failures, timeout exhaustion, host network outage, power loss,
Ollama and participant outcomes remain separate. The cutoff counts encrypted
server traffic across initial tunnels, not artifact payload or total trial
traffic. SDK transfer-byte reuse was not measured. The existing 64 MiB selected
file limit and 120-second per-attempt deadline plus bounded cleanup apply.

## Installed current candidate: 2026-10-14

The [Windows installed-candidate report](evidence/hf-installed-recovery-windows-2026-10-14.json)
repeats the fixed 19,077,344-byte sample with the CI-produced PR #156 wheel.
Its source is `3a8c1ff99cb2c9a873116b0cf1238c11d1688192`, whose tracked tree
equals actual main `3ca8ad0db805ac0c15c9b20d7297e4ce880aa6a2`.
The installed receipt and five service/worker module hashes match the wheel;
the report records the interpreter and SDK versions. No distribution was rebuilt.

Three separate owned database/model/SDK-cache roots passed: early cancellation,
controlled HTTP shutdown after 10 MiB partial output, and forced service process
stop after 10 MiB partial output. The recorded HF descendants exited, restart
preserved each plan and its expected cancelled/failed status, and explicit retry
completed with independently verified final size and SHA-256. Active duplicate
requests retained their original job. All owned services and children stopped;
trial output and previous model files remain available.

The first harness assumed a single descendant; Windows venv launcher trees
contained multiple service/HF processes and the file completed before the gate
was recognized. That attempt does not prove interruption. The corrected harness
identifies only HF worker descendants of its owned process and, for forced stop,
stops the actual service interpreter behind the launcher. Application code was
unchanged. This qualifies these candidate bytes for the named Windows HF paths;
network disconnection, host/power loss, Ollama, inference and participant outcomes
remain separate. SDK byte reuse and aggregate network traffic were not measured.

## Checkout tool and historical evidence

This opt-in checkout tool acquires a public single file through the production
HF runner. It never performs inference or removes files. Normal tests do not
contact model services. Use a fresh empty work directory and a known small
sample; verify its source/license separately before any use.

```powershell
.venv\Scripts\python.exe scripts/verify_hf_recovery.py ggml-org/models-moved tinyllamas/stories15M-q4_0.gguf --revision 499bc8821c6b12b4e53c5bffcb21ec206f212d81 --work-dir .venv/hf-trial-new --output .venv/hf-trial-new.json --max-bytes 67108864 --timeout 120
```

The chosen revision must be a full commit. Metadata must have a positive known
size within the file limit and a SHA-256 digest. Existing nonempty directories
are rejected. The tool isolates databases, destination and HF local auxiliary
cache, disables implicit HF credentials and Xet, and retains files for inspection.
TLS verification remains enabled; a prepared process-only trust bundle may be
needed on a corporate Windows host. Do not disable certificate verification.

The tool checks active duplicate retention, early process cancellation, durable
plan reopening, cancellation after a nonempty SDK `.incomplete` file appears,
then retry and exact size/digest completion. A fast transfer can finish before
partial cancellation: this leaves that gate unproved and returns a failing result.
The timeout applies to each owned download attempt, with bounded cleanup; SDK
metadata uses a request timeout. The file-size limit is not an aggregate network
traffic quota because retries can transfer bytes again.

## Opt-in CI acquisition

The existing CI workflow has a manual `live_hf_recovery` boolean input, disabled
by default. Enabling it runs the fixed public 19 MB sample above in the Linux
and Windows Python 3.12 verification lanes after ordinary tests pass. The sample
identity is not caller-controlled. Each attempt has a 120-second deadline and
64 MiB file limit; the step has a six-minute limit within the ten-minute job.
No inference or model usage occurs. Network failure or a transfer too fast to
observe partial cancellation fails acceptance rather than synthesizing proof.

The JSON-only `hf-recovery-<os>-<source>` artifacts are retained for 30 days;
downloaded model/SDK files are not uploaded. Reports include the CI source SHA
when available. Run `gh workflow run ci.yml --ref EXACT_BRANCH -f live_hf_recovery=true`
only for the source being qualified, and reconcile report identity with the
actual run head. Normal push/PR/manual runs perform no model acquisition.

The [Windows report](evidence/hf-recovery-windows-2026-10-04.json) observed a
10 MiB incomplete file before cancellation and validated the complete 19,077,344
byte artifact. This proves application cancellation/retry and persisted identity;
the SDK owns HTTP resume/cache behavior. Network byte reuse was not measured.
Forced download-service restart, TUI/service independence, other OS/runtime
acceptance and Ollama remain separate evidence gates. The sample's model-card
license declaration was unknown; the trial does not establish usage permission.

The HF runner now consumes child stderr concurrently and retains at most 4096
characters. A local child writing 256 KiB before a nonzero exit proves that pipe
backpressure cannot indefinitely prevent normal terminal-state handling. This
regression runs without network in ordinary tests.

## Forced parent process stop

The service-owned HF child receives a private stdin pipe whose writer stays open
in its parent. A daemon watchdog reads the raw descriptor; EOF or a pipe error
exits the child with status 75, including when SDK/native I/O is blocked. Raw
descriptor reads avoid holding Python's buffered-stdin lock during normal
interpreter shutdown. No PID scan, unrelated process termination or persistent
machine configuration is needed. The parent closes its writer during cleanup.
Legacy direct invocations without the internal guard argument retain their
existing behavior; this boundary protects service-owned HF acquisitions.

Two real local-process regressions cover normal completion and forced parent
termination. A separate [Windows public-file trial](evidence/hf-service-crash-windows-2026-10-04.json)
observed 10 MiB partial data, forcibly stopped the owned service, and observed
the recorded child exit in about 15 ms. Restart marked the abandoned running job
failed, preserved its plan, and an explicit retry completed with exact byte/hash
verification. This was the worktree based on `cf0ca4d`; runtime fingerprints in
the report identify the tested files. It is not released-source, inference,
Ollama or host/power-loss durability proof, and HTTP byte reuse was not measured.
The earlier buffered-stdin implementation failed normal shutdown and the first
trial retry; it was fixed before this passing report. Partial/auxiliary SDK data
remain retained under the existing selected-file deletion policy.
