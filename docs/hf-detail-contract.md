# Hugging Face detail request contract

Detail enrichment uses the configured HF credential and a 10-second SDK request timeout. Credentials are supplied to the client, never returned in the model payload or error text. This is a network request timeout, not a guarantee that UI worker scheduling or every underlying operation finishes in ten seconds.

When the selected result contains a full commit, request that revision and accept only a response whose commit matches. A cache entry is eligible only for the same selected file and commit. Unpinned selections can use the repository cache/main; this remains an explicit unknown or observed identity, not a promise that a mutable branch is stable. Missing selected files and wrong-commit responses cannot replace the selected identity or write new metadata to the durable cache.

`metadata_fetch_status` is `available`, `cached`, `unavailable` or `failed`. A failed request retains existing facts and estimates and adds a fixed public diagnostic. It does not label estimates as newly fetched facts. The artifact declaration has its own availability/partial status; successful fetching does not guarantee byte size, digest, architecture, context support or permission is known. The TUI distinguishes newly fetched details, cached details and unavailable details.

This scope does not infer architecture/KV-cache fields from model names, change numerical scoring, run inference or complete calibration. Those require separately attributed metadata and measured workloads.
