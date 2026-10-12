# Bounded Ollama search metadata concurrency

The Windows participant reported successful startup and search, but slow Ollama
results for `deepsek`. Remote discovery retains the HTML catalog contract in
[ADR 0001](adr/0001-ollama-catalog-source.md); this change does not replace global
discovery with a local installed-model list or change scoring.

Previously, the search response was followed by one serial detail request per
unique catalog model before returning results. Only metadata enrichment now runs
in a pool of at most four workers per search. Each worker owns a private HTTP
Session using the existing retry/adapter policy; all owned Sessions close after
the pool finishes. The shared search Session remains open. Parsing/deduplication
and score/result construction remain sequential; metadata is consumed in catalog
order, including when requests complete out of order. Return limits and
`has_more=False` are unchanged. Metadata transport failure still uses estimated
size, and later parser errors retain already built results.

## Measured evidence

The [2026-10-12 comparison](evidence/ollama-search-latency-windows-2026-10-12.json)
used the old installed `88fed3f` wheel and fixed checkout source in the same
Python 3.12.14 / Requests 2.34.2 interpreter, with TLS verification and separate
empty metadata caches. Both fetched 14 detail pages and returned the same five
result dictionaries, including order, size provenance and scores.

| Implementation | Cold elapsed time | Detail requests | Returned results |
|---|---|---|---|
| Serial installed baseline | 4.62 seconds | 14 | 5 |
| Four-worker fixed source | 1.92 seconds | 14 | 5 |

This is one same-day sample per implementation, not a universal speedup or p95
budget. No competitor implementation was identified or benchmarked. The existing
socket timeouts/retries can still prolong a slow search; four workers are not a
whole-search deadline. There is no added negative metadata cache or deferred
scoring. New installed wheel qualification and participant retest remain separate
from this checkout-source measurement. No model was acquired or executed.

Tests use the existing public search/HTTP boundary: a four-party barrier fails
on the serial implementation, while the corrected path preserves results under
out-of-order completion, confines each client to one worker and closes only its
owned clients. Separate cases pin timeout fallback and partial parser results.
