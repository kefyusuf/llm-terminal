# Hugging Face download identity

## Artifact metadata contract

`artifact_metadata` schema version 1 carries provider, repository, selected file,
resolved commit, upstream LFS SHA-256 when supplied, exact `size_bytes`, source
and model-card links, license declaration and `metadata_observed_at` (Unix UTC
seconds). Missing values are null. `metadata_status` distinguishes available,
partial and unavailable metadata; none of these values is a benchmark or usage
permission. Search metadata may not include exact file size.

Repository model-card data declares `license`, `license_name` and `license_link`,
as described in the [Hub model-card specification](https://huggingface.co/docs/hub/model-cards).
Only HTTPS custom license URLs or files actually present in repository metadata
are linked. The application's MIT license is never substituted for a model's
missing license. Read the linked model terms before use.

Metadata survives detail caching, queue restart and active duplicate requests.
The SQLite migration only adds a nullable JSON column; legacy jobs remain valid
with null metadata. A mismatched declaration is rejected before queue writes;
corrupt/stale persisted declarations are omitted while the job remains visible.
REST model responses and CLI recommendation JSON include the additive field;
the TUI detail screen shows literal metadata with scrolling rather than parsing
upstream license text as Rich markup.

When repository file metadata includes a commit SHA, detail enrichment records
`resolved_revision` alongside the selected `target_file` and caches both values.
A cache entry for another file does not replace an explicit file selection.

New download requests with a known revision require a full 40-character commit
SHA. Branch names and tags are rejected because they can move. The persisted
command carries repository, filename and revision together; the worker passes
that revision to `hf_hub_download`. An active duplicate request retains the
original command. Requeueing a terminal job can select a new revision.

Download-service job JSON includes an additive `artifact` object with
`repository`, `filename`, `revision`, and `revision_status`. Known revisions are
`pinned`; legacy or metadata-free HF requests have a null revision and `unknown`
status. Other providers have a null artifact. Existing SQLite databases need no
schema migration: this view is derived from the persisted command.

## Selected-file path preflight

Queue requests require a portable relative filename with forward-slash folder
separators. Absolute/drive/UNC paths, parent traversal, empty/dot components,
backslashes, control characters, Windows device names, alternate data streams
and ambiguous trailing spaces/dots are rejected before writing the job. Safe
nested filenames are preserved.

Immediately before starting the HF child process, the worker repeats validation
for legacy commands and resolves the selected artifact against the configured
model directory. Existing symlinks or Windows junctions that point outside that
directory are rejected; contained links are accepted. A rejected filename does
not create the model directory. Filesystem errors or a directory where the file
should be produce a terminal failed job without launching a download child.
Error details use fixed text rather than exposing absolute local paths.

This checks the selected artifact path at that moment. It is not an OS sandbox:
concurrent filesystem changes after the check and SDK-managed auxiliary cache
paths are outside this guard. The canonical configured root is passed to the
child. Roots are not isolated by repository or revision yet.

## Limits and remaining release gates

- Legacy requests remain supported and download the SDK's default revision.
  Unknown metadata is not an immutable identity guarantee.
- Cached identity describes the previously observed commit, not necessarily the
  latest repository head. This does not independently verify a file digest.
- Queue deduplication remains repository-based. Different files or revisions of
  one repository cannot run as separate active jobs yet.
- Repository/revision destination isolation, auxiliary-cache containment,
  byte/disk preflight, license/model-card metadata and user-facing confirmation
  remain pending in roadmap M1-D1.
- Tests exercise enrichment, cached metadata, SQLite restart/duplicate/requeue,
  the worker's child arguments and the generated SDK call without downloading
  model weights. A real bounded download/cancel/recovery trial remains M1-D2.

This is the revision-pinning slice of the proposed roadmap in PR #118, not
completion of all artifact preflight or evidence of production readiness.
