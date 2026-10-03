# Shareable diagnostics

Run an explicit diagnostic without starting runtimes, downloading models, or changing trust settings:

```console
ai-model-explorer-cli doctor
ai-model-explorer-cli doctor --json --timeout 5
ai-model-explorer-cli doctor --offline --json
```

From a checkout, use `python cli.py doctor` with the project's Python environment. The command is included in the existing installed CLI entry point.

## Checks and interpretation

The report contains application/Python versions, HF token presence, the configured CA policy, model/cache/download-history path existence and type, and free space on their existing ancestor volumes. Paths and addresses are represented by fixed labels. These checks do not write files, prove write permission, or calculate whether a particular model fits on disk. Missing storage locations are normal before first use.

Online mode concurrently checks the configured Ollama `/api/version`, LM Studio `/v1/models`, Docker Model Runner `/models`, download-service `/health`, and the public Hugging Face landing endpoint. It respects `AIMODEL_OLLAMA_API_BASE`, `LMSTUDIO_HOST`, `DOCKER_MODEL_RUNNER_HOST`, `HF_ENDPOINT`, and the existing loopback-only download-service configuration. HF requires HTTPS. Endpoint credentials, query parameters, fragments, malformed ports, and non-HTTP(S) schemes are rejected. The public HF request sends no HF token; token presence does not prove authentication or gated-model access.

HTTP 200 means the configured endpoint responded. It does not prove inference, model availability, API compatibility, or download-service health beyond reachability. An allowlisted numeric Ollama version is included when its bounded response can be parsed; LM Studio and Docker versions are not inferred from model listings. MLX reports platform eligibility and optional package metadata without importing or running MLX. Installation metadata does not prove MLX inference works.

The default **three-second total network wait budget** is shared by all concurrent probes. `--timeout` accepts values greater than zero and up to 30 seconds. A probe that has not completed becomes `deadline`; daemon probes do not hold CLI process exit. Requests also have individual timeouts, no retry adapter, and no redirects. The total command duration additionally includes Python imports and local filesystem/CA checks, which are outside the network budget. This command is intended for one-shot CLI use, not recurring polling inside a long-lived process.

`--offline` skips network checks explicitly; an offline `ok` result is not connectivity evidence. Optional runtimes or the download service being unreachable produce warnings and exit **0**. HF network failure, TLS failures, invalid CA/endpoint configuration, and inaccessible or incorrectly typed storage produce errors and exit **1**. Click argument errors exit **2**. JSON schema version is `1`; each check has a fixed `name`, `status`, `code`, and `action`, with selected safe factual fields. JSON stdout remains parseable even when the report exits 1.

## TLS troubleshooting

TLS verification remains enabled. Requests uses its default trust bundle or `REQUESTS_CA_BUNDLE` (falling back to `CURL_CA_BUNDLE`). A custom PEM bundle must be readable and parseable. A CA directory must be usable by OpenSSL; the local check cannot prove that its hashed certificates trust a specific server. A successful HTTPS probe supplies that separate evidence.

For an organization proxy or private CA, obtain the approved trust chain and point `REQUESTS_CA_BUNDLE` to its PEM bundle before rerunning. Do not disable verification. The diagnostic does not export Windows trust, install certificates, or modify machine trust. Standard Requests proxy environment settings remain in effect.

## Privacy and limits

Both human and JSON reports omit token values, paths, endpoint URLs, usernames, headers, certificate subjects, model names, response bodies, and exception text. Requests' implicit `.netrc` authentication is disabled for these diagnostic requests. Runtime version output accepts only bounded numeric version strings. The report is an allowlist, rather than a best-effort regex redaction of arbitrary messages.

Like existing CLI commands, `doctor` requires settings to load successfully first. A Pydantic configuration error during CLI import occurs before this reporting boundary; its traceback is not a shareable doctor report. Correct that configuration locally before using `doctor --json`. Network intermediaries can observe the explicit probes, and configured proxies can use their own credentials.

## Verification

Ten regression tests cover offline behavior, token/path redaction, nonblocking deadline expiry, TLS actions without exception leakage, TLS/redirect/auth settings, invalid CA files, invalid endpoints, HTTPS enforcement, optional runtime failure, and JSON error exit behavior.

The native Windows checkout was also exercised with `doctor --timeout 3`: HF and the existing download service responded, optional runtime probes reached the shared deadline, and the command returned a warning with exit 0 in about 3.7 seconds including imports. No runtimes were started. This is a diagnostic smoke result, not a provider or inference certification.

This implements the network-diagnostics portion of roadmap R2 proposed in PR #118. Estimate provenance and user-facing estimate labels remain separate work.
