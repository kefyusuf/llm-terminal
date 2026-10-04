# Qualified candidate restore rehearsal

The [Windows report](evidence/qualified-candidate-restore-windows-2026-10-04.json)
extends the earlier [read-only SQLite qualification](sqlite-compatibility-rehearsal.md)
with actual installed worker execution and selected-file restoration. This is an
isolated rehearsal, not authorization for production downgrade.

The current candidate is `6bab5a6488749e97ef6f7679fa280e5335c3ad84` from
[Package run 37232846239](https://github.com/kefyusuf/llm-terminal/actions/runs/37232846239).
The previous candidate is `5d65b5fcaf0997290e7df9fcdf4dd5df31ed334c` from
[Package run 37229916321](https://github.com/kefyusuf/llm-terminal/actions/runs/37229916321).
Both runs passed all 19 jobs. Unlike the earlier read-only target `cf0ca4d`,
this previous worker includes the child-lifetime guard. Both development
packages report version 1.0.1; the distinct wheel hashes in the report identify
the actual tested bytes.

## Procedure and observed result

1. Download each CI-produced wheel/sdist pair and manifest. Verify each complete
   source identity and distribution hashes. Install each wheel in a separate
   fresh Python 3.12.14 environment and require `pip check` success. Verify
   installed module ownership and the PEP 610 wheel receipt against its manifest.
2. Start the current installed service with `python -I -m downloads.download_service`
   from a fresh trial directory, removed `PYTHONPATH`, isolated database/cache/model
   roots, loopback port and private process-only token. Keep TLS enabled and
   disable implicit HF token use. Only the owned service process is controlled.
3. Request `/jobs/plan` for the pinned public 19,077,344-byte sample. Resolve and
   verify its destination is inside the trial model root before restoring the
   previously hash-verified selected file. Submit `/jobs`, wait at most 120 seconds
   for completion, and verify the selected file size and SHA-256 again. Gracefully
   stop the service and require exit code zero.
4. Create a terminal-job snapshot with SQLite's backup API and `integrity_check`.
   Preserve the selected model file separately. Record both hashes. Clone the
   database into a second quarantine root before starting the previous service.
5. Confirm the previous service reads the completed job. Its restored record still
   refers to the original root, so keep it terminal. Request a fresh server-owned
   plan, verify its destination belongs to the second root, restore the selected
   file there, then explicitly submit the job. The server recomputes and persists
   the new plan before the worker claims the job. Verify completed status, file
   size/digest and controlled shutdown exit zero.
6. Verify that the immutable database/file backups, current candidate's selected
   file and original trial file retain their hashes.

Every step passed. The database contained one terminal job; both installed
workers completed the acquisition and verified the same pinned file. Services
used separate roots and were stopped before the next phase. The report excludes
local paths, ports, tokens and private model identities.

This does not qualify arbitrary older workers, active-job migration, shared
destinations, auxiliary cache restoration, host/power-loss recovery, inference,
usage permission, transfer byte reuse, release withdrawal or production rollback.
The selected public file was restored locally; SDK requests may still access the
network. Never infer zero transfer bytes from the completed status.

## Hosted acquisition evidence

[CI run 37232844033](https://github.com/kefyusuf/llm-terminal/actions/runs/37232844033)
passed all 11 jobs at source `6bab5a6`, including the manually selected live HF
gate on Linux and Windows Python 3.12. Both archived reports observed 10 MiB of
partial output, cancellation, reopen with preserved plan, retry completion and
exact final size/digest. The [Linux report](evidence/hf-recovery-ci-linux-2026-10-04.json)
and [Windows report](evidence/hf-recovery-ci-windows-2026-10-04.json) retain their
full CI source identity. This gate uses a fixed public sample and is disabled by
default; it does not certify GPU inference or every HF artifact.

The integrated candidate `ca86b0d` was requalified after the claimed-job
cancel/delete fix in [live CI run 37236560256](https://github.com/kefyusuf/llm-terminal/actions/runs/37236560256),
all 11 jobs passed. [Linux](evidence/hf-recovery-integration-linux-2026-10-05.json)
observed 4 MiB and [Windows](evidence/hf-recovery-integration-windows-2026-10-05.json)
10 MiB partial output before cancellation; both reopened/retried and verified
the same pinned final bytes/digest. This candidate's tracked file tree exactly
matches actual squash-merged main source `93395de`, which independently passed
[main CI, 11 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37236972922)
and [main Package, 19 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37236975704).
The earlier previous-worker restore report retains its original named candidates;
the new integration qualification is not a new production rollback rehearsal.

## Actual main candidate requalification

The [2026-10-05 Windows report](evidence/qualified-main-restore-windows-2026-10-05.json)
repeats the installed-worker procedure with current main source
`eabf63e5651bc9bd2e07fd9ad0fbaee80f16179c` and previous qualified source
`6bab5a6488749e97ef6f7679fa280e5335c3ad84`. Current main passed
[CI 11/11](https://github.com/kefyusuf/llm-terminal/actions/runs/37240529606)
and [Package 19/19](https://github.com/kefyusuf/llm-terminal/actions/runs/37242705905).
The rehearsal consumed the preserved CI wheel/sdist pair and manifest without
rebuilding. Its wheel SHA-256 is
`b17695bde1b8a76e7220ade88271af4faa5fe7f094e34a2646ce9e4ad84890ab`;
the previous wheel remains identified by its original manifest and report.

Both candidate environments passed `pip check`. The current wheel was installed
in a new isolated Python 3.12.14 environment. Installed origins and PEP 610 wheel
receipts matched their manifests. Both actual workers completed with verified
19,077,344-byte pinned output and exited zero after controlled shutdown. The
previous worker read the one terminal record from the current candidate's SQLite
backup, then recomputed its plan for a separate quarantine root before requeue.
The database backup passed integrity checking; immutable backups and original
selected files retained their hashes. No original database was used by the
previous worker.

This closes the named-main candidate's bounded terminal-job/file restore check.
The package version is still 1.0.1; an authorized new release version must be
qualified separately. Active-job migration, arbitrary downgrade safety, shared
destinations, auxiliary caches, power loss, inference, transfer reuse and
production rollback remain outside this evidence.
