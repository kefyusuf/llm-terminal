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

This selected-file check remains the compatibility path for legacy unplanned
jobs. New service requests use the managed plan below.

## Managed download plans

Authenticated `POST /jobs/plan` previews an HF selection without queue writes or
model-directory creation. `POST /jobs` recalculates the authoritative plan inside
a SQLite write transaction. Each job stores its bound plan through restart.
Destinations are `<model-root>/huggingface/<sha256(repository)>/<commit-or-unresolved>/<file>`;
the full repository hash separates repository namespaces. Active duplicate
requests preserve their original metadata and destination.

Known queued/running byte counts reserve their full size, even after partial
transfer. A plan requires artifact bytes plus other known reservations and a
64 MiB safety margin. Unknown-sized jobs cannot reserve an unknown byte count;
they require explicit `allow_unknown_size: true`. The TUI labels this exception
before its download action. Disk snapshots can change and the margin is not a
guarantee against disk exhaustion.

Existing managed trees are bounded to 4096 entries and checked for symlinks,
junctions and hardlinked files, including SDK auxiliary directories. The worker
rechecks the plan, current disk budget and configured root before launch. A
changed root, corrupt persisted plan or unsafe tree fails without a child.
Concurrent filesystem changes after a check remain outside this guard: this is
not an OS sandbox. Legacy unplanned jobs retain their selected-file guard.

The SDK child requires a contained regular selected file after download and
matches known byte count and upstream SHA-256 before reporting completion.
This validates transfer identity, not model format or inference compatibility.

`ai-model-explorer-cli download-plan REPOSITORY FILENAME --json` obtains current upstream
metadata and emits schema version 1 with `allowed`, `status`, warnings, declared
metadata and local paths. It does not enqueue or download. A successful JSON
command can contain a blocked plan; automation must inspect `allowed`. Offline
mode leaves unknown facts explicit; a requested full commit records requested
identity rather than claiming remote verification. Paths in this private plan
are not the redacted output of `doctor`.

The copied command selects one exact file and commit using the current Python
environment's HF CLI module. Shell quoting supports PowerShell and POSIX shells.
Manual execution bypasses the application's queue reservations and completion
guard; inspect the plan and license before using it.

Service protocol 2.0 is required for plan-aware clients. Automatic upgrade does
not stop an incompatible service with active jobs or uncertain job history.
Graceful shutdown addresses only the configured service; fallback force-stop
is limited to the process launched by this client, never a system-wide scan.

## Limits and remaining release gates

- Legacy requests remain supported and download the SDK's default revision.
  Unknown metadata is not an immutable identity guarantee.
- Cached identity describes the previously observed commit, not necessarily the
  latest repository head. Unknown digests cannot be verified independently.
- Queue deduplication remains repository-based. Different files or revisions of
  one repository cannot run as separate active jobs yet.
- Tests exercise enrichment, cached metadata, SQLite restart/duplicate/requeue,
  the worker's child arguments and the generated SDK call without downloading
  model weights. The separate [Windows real SDK trial](evidence/hf-download-windows-2026-10-04.json)
  verified early cancellation, reopen, retry and exact bytes/SHA-256. The later
  [reproducible recovery trial](hf-recovery-acceptance.md) also observed partial
  cancellation and successful retry. Forced service restart and Ollama remain
  separate M1-D2 gates.

These are artifact identity and preflight slices of the proposed roadmap in
PR #118. These checks and bounded trials do not establish production readiness.
