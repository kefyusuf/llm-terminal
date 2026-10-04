# Versioned model exports and offline comparison

`ai-model-explorer-cli search QUERY --provider huggingface --json` adds a schema-1
object without changing the existing search table or `recommend --json` array.
Fields include `kind: search`, query/provider/limit/sort, selected model records
and legacy provider error strings. Partial models and errors can coexist; a
successful command is not proof that every provider succeeded. Warnings remain
on stderr, while stdout is one valid JSON object without Rich status/wrapping.

Each record selects public identity/name/source, parameter/quantization/size
labels, selected filename/revision, scores, score provenance and bound artifact
metadata. Unknown fields are null. Invalid bound artifact metadata is omitted
with `artifact_metadata_status: invalid`; no application license or measured
score is invented. The export does not contain private destination plans or
application settings. Model identities and upstream metadata can themselves be
private; this is not the redacted `doctor` contract.

```console
ai-model-explorer-cli search qwen --provider huggingface --limit 10 --json > search.json
ai-model-explorer-cli compare --input search.json owner/model-one owner/model-two --json
ai-model-explorer-cli plan model-7b --context 8192 --json
```

Comparison selects between two and eight distinct IDs from the saved schema-1
search in the requested order. It preserves scores, provenance, selected facts
and original error strings without new hardware detection, provider calls or
rescoring. Missing/ambiguous IDs and unsupported schemas/types are rejected.
Input is bounded to 2 MiB. Literal names remain literal in JSON and comparison
tables. This lets automation compare a recorded selection without UI scraping.

Hardware-plan JSON has `kind: hardware_plan`, model name, positive requested
context, existing quantization plans, null artifact metadata and explicit
heuristic provenance. It does not establish supported token capacity, runtime
allocation or observed throughput. Legacy context overhead/offload formulas and
numeric values are unchanged; `quality_rank` remains a quantization proxy, not a
universal quality rank. The legacy `total_mem_needed` offload field is not an
independently measured combined RAM/VRAM requirement. Metadata-aware memory
planning/calibration remains a separate scope.

Exports reject non-finite/unsupported JSON values before stdout is written.
Consumers must check `schema_version`, `kind`, errors and metadata availability;
additive fields can appear later. This schema does not promise a download,
runtime load or inference outcome. Installed package consumers exercise the
hardware-plan and saved-comparison JSON commands outside the checkout.

## Exact artifact handoff

Use `download-plan REPOSITORY FILENAME --json` to obtain the selected destination
and known disk/identity/license facts; inspect `allowed` and warnings. Download
completion can validate bytes/digest but does not establish inference format or
model usage permission. Keep the original identity/provenance with any handoff.

For a prepared Ollama runtime, its official [GGUF import procedure](https://docs.ollama.com/import)
uses a `Modelfile` whose `FROM` points to the exact selected file, then an explicit
`ollama create` with a local model name. A Modelfile beside the artifact can use
`FROM ./selected.gguf`, replacing that basename with the selected file's actual
name. Do not substitute a quantization wildcard or silently invoke a runtime
from the exporter. The documented command is a handoff, not a tested import on
this host; inspect upstream license terms and the support matrix before use.
