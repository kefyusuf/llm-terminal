# Managed selected-file removal and service shutdown

Service protocol 2.1 adds authenticated `POST /jobs/delete` with an optional
boolean `delete_data`. Omitting it retains record-only deletion. The service
derives the selected path from persisted HF artifact identity and the configured
model root; caller-supplied local paths cannot select files to delete.

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
Old services cannot silently ignore the new flag: the client requires protocol
2.1, preserving the existing protection against upgrading an active service.
Ollama's explicit removal still goes through its CLI and has separate runtime
acceptance requirements.

Controlled service shutdown requests cancellation of active owned children,
preserves unclaimed queued jobs and waits for the worker dispatcher. Workers
also observe service stop events to contain claim/registration races. Launching
with `python -m downloads.download_service` now shares the same module state
with the API handler; otherwise relative imports could create a second state
whose shutdown call never reached the actual server.

The [native Windows trial](evidence/service-restart-windows-2026-10-04.json)
started an isolated authenticated service, queued the real pinned 19 MB sample,
stopped it while running, verified a cancelled job, restarted the service,
preserved the plan, retried to exact bytes/SHA-256 completion and removed only
the selected file while retaining a sibling. A real local process regression
also proves that module-launched HTTP shutdown exits without any model download.
Forced process/host crashes, other OS live acquisition, Ollama and inference
remain separate gates. Retained SDK partial files require deliberate inspection,
not a claim that all cache data was deleted.
