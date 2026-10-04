# Bounded Hugging Face recovery acceptance

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
