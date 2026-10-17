# Managed selected-file removal and download shutdown

## Application ownership

The TUI owns one `downloads.manager.DownloadManager`. Bounded worker threads run
HF transfers and Ollama streaming pulls inside the application process. The
`AIMODEL_DOWNLOAD_MAX_WORKERS` setting controls parallelism. Later jobs remain
queued until a worker is available.

Downloads stop when the app exits. `shutdown()` stops admission, requests
cancellation and joins workers. History remains in the existing SQLite database.
Startup marks abandoned queued, running and canceling records `interrupted`;
it does not automatically restart network transfers. Submitting an interrupted
job again resumes retained HF SDK partial files or Ollama's existing pull data.
There is no separate download server, port, bearer token or ownership sidecar.

Legacy ID normalization retains the newest record by `updated_at`, then the
highest row ID for ties. It holds one immediate SQLite transaction across the
read and all mutations, removes duplicate loser records before normalizing the
winner ID, and preserves the winner's other fields. An update failure rolls back
the duplicate removals as well. Only duplicate database records are removed;
model files are untouched. Repeating normalization is idempotent.

`python scripts/dev.py smoke` checks an isolated manager lifecycle without
downloading models. Deterministic fake-backend tests cover progress, duplicate
submission, cancellation, shutdown and interrupted startup recovery. These
checks do not qualify live-provider downloads.

## Removal and cancellation

`DownloadManager.delete_job(target_id, delete_data=False)` removes history only
by default. When `delete_data=True`, the store derives the selected path from
persisted HF artifact identity and the configured model root; caller-supplied
local paths cannot select files to delete.

Data removal requires a terminal job and a matching managed plan. Active jobs,
legacy/unplanned jobs, corrupt or changed paths and linked/hardlinked managed
trees are rejected without removing the record or data. Deletion holds a SQLite
write transaction against concurrent requeue. Only the exact selected file is
unlinked; other files, older variants, SDK auxiliary/partial files and shared
caches remain. No directory is recursively deleted. Filesystem deletion and the
database commit are not one atomic operation; retry after a missing selected
file can safely clear the remaining record.

The TUI's HF action is labeled **Delete File**. Cancellation must be confirmed
before requesting data removal; a pending/uncertain cancellation preserves data.
Queued cancellation can terminalize a job immediately. A running job remains
active until its worker acknowledges cancellation. Manager locking and per-job
cancellation events prevent concurrent same-target submission from scheduling
duplicate transfers, including cancellation followed by immediate resubmission.
Ollama's explicit model removal still goes through its CLI and has separate
runtime acceptance requirements.

## Historical evidence

The [native Windows trial](evidence/service-restart-windows-2026-10-04.json)
tested the former authenticated download service. It queued the real pinned
19 MB sample, stopped it while running, verified cancellation, restarted the
service, preserved the plan, retried to exact bytes/SHA-256 completion and
removed only the selected file while retaining a sibling. The former service's
socket ownership, process shutdown and protocol tests applied to that retired
implementation. This evidence does not qualify the in-process manager.

Forced process/host crashes, other OS live acquisition, Ollama and inference
remain separate gates. Retained SDK partial files require deliberate inspection,
not a claim that all cache data was deleted.
