# ADR 0001: Separate remote Ollama catalog discovery from local model facts

Date: 2026-10-04. Status: accepted for the current implementation boundary.

## Context

Remote discovery currently parses `ollama.com` HTML. Fixtures cover known pages,
and unsupported successful HTTP page shapes emit `parse_error` and a diagnostic
instead of claiming a successful empty catalog. This remains sensitive to
upstream layout changes and must not be confused with local runtime discovery.

The official [API index](https://docs.ollama.com/llms.txt) describes installed and
running model operations. [GET /api/tags](https://docs.ollama.com/api/tags) lists
available models on the addressed instance; [POST /api/show](https://docs.ollama.com/api-reference/show-model-details)
can supply that instance's model details, capability and model-info declarations.
No supported global registry-search replacement was established in this review.
This is a bounded conclusion from the reviewed API surface, not proof that no
future catalog API can exist.

## Decision

Retain fixture-backed remote HTML discovery and observable shape failures.
Do not replace a global search with `/api/tags`, which changes the user's result
universe to already installed models. Use local structured details only when a
configured runtime is reachable and the requested model exists there, with
explicit instance/model provenance and unknown fields on failure. This ADR does
not claim a newly implemented local enrichment or prepared-runtime acceptance.

A future source migration requires a supported catalog contract, equivalent
search/pagination semantics, authentication/deadline rules and fixtures before
changing the provider's boundary. It is conditional work, not a release gate
that can be satisfied by inventing an undocumented endpoint.

## Drift response

Distinguish HTTP/transport failures from parser-shape errors and valid no-result
responses. On `unsupported search page shape`, retain partial provider results,
identify the source and error code, compare a redacted failing page to fixtures
and add a representative regression before changing the parser. Do not hide an
outage as an empty result or downgrade deterministic test gates. The existing
daily live-search workflow remains separate from offline merge verification;
its future runs are not presumed green from a prior fixture result.
