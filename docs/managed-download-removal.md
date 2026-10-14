# Managed selected-file removal and service shutdown

## Startup ownership

Normal and smoke service startup acquire an exclusive SQLite transaction on a
separate `<resolved-download-db>.service-lock` file, then bind the HTTP socket.
Only after both succeed does the service initialize/migrate the job store,
recover abandoned running jobs and start its worker. Importing the service
does not initialize or recover the store. A second process using the same
database is rejected even when it selects another port or an ephemeral smoke
port; independent databases can run separately.

The ownership connection stays open until started workers finish. Failed binds
and thread starts close their acquired resources. Process death releases SQLite
ownership so a replacement can perform the existing abandoned-job recovery.
The sidecar is not the jobs database, so normal queue/API writes remain usable.
Keep the sidecar file: removing it while a process owns it can break exclusivity.
No download status rules, queue capacity, model destinations or protocol version
change.

Legacy ID normalization retains the newest record by `updated_at`, then the
highest row ID for ties. It holds one immediate SQLite transaction across the
read and all mutations, removes duplicate loser records before normalizing the
winner ID, and preserves the winner's other fields. An update failure rolls back
the duplicate removals as well. Only duplicate database records are removed;
model files are untouched. Repeating normalization is idempotent.

Real-process regressions cover an occupied socket, a second normal/smoke process
on the same database, owner death/recovery and independent databases. An injected
thread-start failure verifies that the smoke worker stops before ownership is
released. These tests do not acquire models or qualify live-provider downloads.

`python scripts/dev.py smoke` uses a temporary, separate download database for
its service check so an existing development service is preserved. Direct
service smoke startup retains the same ownership rule as normal startup.

## Removal and shutdown

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

The dispatcher claims a job only
when execution capacity is available. A saturated pool leaves later jobs queued,
so they can be cancelled and their records removed immediately. A completed
worker frees a slot without waiting for the other workers. Shutdown preserves
unclaimed queued jobs. Real-service tests exercise these paths with harmless
controlled child processes; no model acquisition is implied.

Queued cancellation and worker claim are serialized by the store. Cancellation
can terminalize an unclaimed queued job immediately; a claimed running job stays
active even before its child process is registered. Only the worker can acknowledge
that cancellation as terminal. Both HF and streamed workers check cancellation
before launch and treat a missing record as cancelled. Deletion/requeue therefore
cannot race an unregistered claimed worker into an invisible new download.
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
