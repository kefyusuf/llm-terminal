# Score estimates and provenance

Quality, speed, fit, context and composite scores are **heuristic ranking aids**. They are not measured benchmark results, probabilities, accuracy percentages, verified runtime throughput or supported context lengths. This change preserves the existing formulas, numeric fields, ordering and use-case weights.

## What each estimate uses

| Dimension | Current basis | Limits |
| --- | --- | --- |
| Quality | Parameter-count logarithm plus quantization rank | Does not measure task performance, training quality or benchmark accuracy. Missing/unrecognized quantization uses the existing default contribution. |
| Speed / estimated tok/s | GPU bandwidth lookup or backend default, divided by supplied model size, multiplied by a mode efficiency factor | No runtime benchmark. Context length, batching, cache, offload and scheduling are unmeasured. CPU mode still uses the existing GPU/backend bandwidth assumption with CPU efficiency. |
| Fit score | Model-size utilization of total VRAM/RAM and supplied mode | Does not use current free memory or include KV cache/runtime overhead in this score. Separate fit classification may use different inputs. |
| Context | Model-size thresholds | A legacy size proxy, not a measured context window or verified token capacity. Model size does not establish supported context length. |
| Composite | Use-case-weighted heuristic dimensions | A relative ranking under those assumptions, not a universal benchmark. Unknown use cases retain the general weights. |

The bandwidth table is a static repository lookup with existing substring/token matching. A lookup hit is not proof of exact GPU identification or measured bandwidth. Unknown GPUs use a backend default; unknown backends retain the CUDA default. Unknown modes retain the legacy GPU efficiency fallback. `No Fit` or nonpositive size suppresses throughput computation.

## Additive machine-readable contract

`score_model()` returns the same numeric `Scores` fields plus `provenance`. Enriched model dictionaries include `score_provenance`. REST model lists, REST `/api/v1/scores/{model}`, and CLI `recommend --json` expose `score_provenance` beside the existing `scores` object. The recommendation output remains a JSON array; public numeric field names are unchanged.

Metadata schema version `1` includes:

- `kind: heuristic`, `measured: false`, and the supplied `size_gb`.
- Per-dimension `basis` and limitations; quality parameter/quantization inputs, fit memory totals and mode, and composite weight selection.
- Speed `bandwidth_source` (`gpu_lookup`, `backend_default`, `cuda_default`), bandwidth, efficiency, default-mode flag, and `computed` flag.
- `size_source`: `supplied_size_gb`, `display_size_parse`, or `model_name_estimate` for the dedicated REST scoring endpoint. These identify the scoring input route, not an independently verified model artifact size. Re-enrichment preserves the route while that size is unchanged.

Legacy/cached model rows without provenance are serialized with `score_provenance: null`. No metadata is invented for those rows and no rescoring is forced merely to populate it. Fresh enrichment adds metadata. TUI comparison shows `Unknown` for a missing bandwidth basis.

## Visible explanations

CLI `scores MODEL` shows the heuristic notice, each dimension's limitations and the actual bandwidth source. Recommendation tables are labeled as heuristic estimates. TUI comparison shows a persistent explanation of parameter/quantization proxies, unmeasured throughput, total-memory fit and the context-size proxy, plus each model's bandwidth basis. Existing popularity/gem scores are separate and keep their existing labels.

This addresses the estimate-explanation portion of roadmap R2 proposed in PR #118, following the diagnostic command in PR #122. It does not validate the accuracy of legacy formulas or replace them with benchmark data. Future formula changes need separate acceptance criteria and validation.

## Verification

Regression tests reproduce missing provenance at the core, REST and CLI boundaries and missing explanations in a mounted Textual comparison screen. They pin known numeric outputs, lookup/default/no-fit behavior, missing legacy metadata, weight fallback and size-source preservation across repeated enrichment. Tests run without remote model requests or inference.

Recommendation JSON is emitted directly through Click so terminal wrapping cannot insert newlines inside metadata strings. A narrow-terminal regression reproduced that failure before the fix.
