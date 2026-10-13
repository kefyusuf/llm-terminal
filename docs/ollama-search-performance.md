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

## Early visible results and optional local probe (2026-10-13)

Single-provider Ollama searches now publish deep-copied catalog-order prefixes
after metadata enrichment and scoring, up to the visible result limit. The TUI
shows these rows while the remaining search continues. Only final results enter
the query cache. Search IDs reject obsolete work, and table selection follows the
model row key when sorting inserts a new row before it. All-provider searches,
final ordering, metadata and scores retain their existing behavior.

The optional `/api/tags` lookup used the shared retry policy and took 4.04 seconds
against an unavailable local endpoint despite a one-second request timeout. It
now owns a one-attempt Session that closes on every exit. Remote clients retain
their two retries. This does not impose a whole-search deadline.

The [live headless observation](evidence/ollama-early-results-windows-2026-10-13.json)
used `deepsek`, an empty metadata cache, preinitialized hardware and an unavailable
local runtime. Before the probe correction the first visible row took 4.98 seconds
and completion 6.19 seconds; afterward these were 3.03 and 4.25 seconds. Both
returned 14 results without an error. These are separate single live-network
observations, not a controlled latency benchmark, participant feedback or p95.
Periodic discovery/download polling was disabled in the minimal test mount;
normal search, debounce, filtering and table rendering ran. No model was acquired
or executed. The participant's earlier qualitative feedback was only that the
previous build felt somewhat better; this build still needs participant retest.

Headless regression tests hold a later metadata request open and verify visible
scored rows before completion, no early cache entry, stable selection after
sorting, final cache population and stale-query rejection. Provider tests verify
snapshot isolation, the preview limit and suppression for combined searches.
A real local HTTP server verifies one failed probe attempt and unchanged
installed-name normalization on success.
