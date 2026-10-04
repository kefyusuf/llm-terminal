# Bounded local runtime calibration

Run `scripts/calibrate_ollama.py` only against a prepared dedicated local Ollama with an already installed exact model name. The tool never pulls, imports, deletes or publishes models. It accepts credential-free loopback HTTP, disables proxies/redirects, and requires `/api/status` to explicitly report cloud disabled. Runtime versions without that experimental status contract remain unsupported for this tool. Remote model declarations/cloud tags are rejected before generation. See the official [cloud guidance](https://docs.ollama.com/cloud), [API client status implementation](https://github.com/ollama/ollama/blob/main/api/client.go) and [remote-model fields](https://github.com/ollama/ollama/blob/main/api/types.go). No API key is sent.

```powershell
.venv\Scripts\python.exe scripts/calibrate_ollama.py exact-installed-name:tag --url http://127.0.0.1:11434 --repetitions 3 --tokens 64 --context 512 --max-seconds 120 --request-timeout 30 --output .venv/calibration-new.json
```

Repetitions are bounded to 2–10 plus one excluded warmup; requested tokens to 1–128, context to 1–8192, request timeout to 1–30 seconds and whole worker budget to 5–300 seconds. The supervising client terminates only its owned worker after the budget plus two seconds. A socket timeout is not an overall deadline by itself. Client termination/connection closure is not proof that the separately owned Ollama daemon immediately stopped work; do not run this on a shared runtime. The model keep-alive request is 30 seconds. No persistent runtime configuration is changed.

The fixed public `hash-explanation-v1` workload uses deterministic request settings and non-streaming output. A completion capability is required; thinking must be verifiably disabled when advertised. Actual generated counts must respect the requested token limit. Requested context above declared support is refused; unknown supported context remains unknown. Exact digest and runtime version are checked before and after the sample set. These observations are not cryptographic runtime attestations.

Reports retain CPU/GPU/RAM observations, runtime/version, selected model name/digest and filtered declarations, workload hash/settings, warmup and sample metrics, medians, min/max and population standard deviation. Generated response/thinking text, licenses, credentials and arbitrary input prompts are omitted. Selected identities/hardware can be private; this is not a redacted doctor report. The output must be fresh and is written only after successful completion.

[Ollama generate](https://docs.ollama.com/api/generate) supplies token counts and nanosecond timing fields. Generation rate, prompt rate, runtime total/load duration and measured HTTP wall time remain separate. When a cached prompt count is reported, prompt throughput uses count minus cached count; absent cache counts remain explicitly unreported. Zero prompt evaluation duration yields unknown prompt throughput. No first-token latency is inferred from non-streaming output.

Prediction uses the existing bandwidth heuristic with runtime disk size as an explicitly labeled proxy and a fit-derived, estimated mode. Actual offload/allocation is not measured. Signed error on median generation throughput is `(prediction - observed median) / observed median * 100`; positive means overprediction. A sequential workload and a few repetitions do not establish general model-quality ranking, concurrency scaling or universal accuracy. No scoring coefficient is changed.

HTTP sample duration uses `time.perf_counter()` independently from the monotonic
deadline clock. The native Windows Python 3.12.14 runtime reported a 15.625 ms
`GetTickCount64` resolution for `monotonic`, which can return identical ticks
around a fast local response; its `QueryPerformanceCounter` resolution was
100 ns. A deterministic coarse-clock regression verifies that this does not
invalidate a successful sample or bypass the subsequent identity check.

The schema-1 report's `model_facts` can feed [saved-facts memory scenarios](metadata-memory-scenarios.md). Archive actual reports by hardware/runtime/model digest and workload before evaluating prediction errors across a real corpus. CI tests use simulated metrics and a real local HTTP fixture to verify request/JSON/deadline boundaries; they are not inference measurements. No genuine Ollama benchmark has been collected on the current host because a prepared runtime/model is unavailable.
