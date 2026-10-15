# Verified continuation record

Date: 2026-10-16. Repository: `kefyusuf/llm-terminal`.

## Current narrow continuation

PR #171 is merged at `2267ad7f8165809ba0123807026896866b2bc2aa`.
PR/main CI passed 11/11, Package 19/19 with 18 matching consumer reports.
The qualified pilot passed 19 installed feedback/response cases, 62 module hashes
and startup/owned cleanup. Candidate and main Git trees match; details are in
`.venv/admission-response-completed-1016.json`. Previous environments/models and
user ignore files remain preserved.

Branch `fix/download-admission-response-loss` covers server acceptance followed
by missing/corrupt/timed-out response. Real HTTP admission committed a queued job
but the UI reported failure. Expected transport/body errors now report unknown
outcome and direct the user to history before retrying, without resubmission.
Four real store/HTTP cases cover connection loss, truncated response, invalid
JSON and timeout; each preserves local state, proves exactly one accepted POST
and recovers the actual queued job through history/synchronization. Together
with existing admission/manager/sync/startup cases, 67 tests pass; Ruff and
38-file mypy passed. Check full verify, review, PR CI, Package and actual-main
CI before completion or pilot refresh. No worker, protocol, storage or automatic
retry changes are in scope.

Remaining product acceptance is distinct from code TODOs: participant pilot,
live provider/network download/recovery and power-loss evidence remain open.
Release version/account/name ownership, protected publishing approval and
qualification of the actual release/rollback byte pair remain separate gates.
Do not reopen completed roadmap features or publish/promote from development
candidate evidence. See `.planning/roadmap.md` and `docs/release-readiness.md`.

Earlier checkpoints below are historical.

PR #170 is merged at `03e68d55937272b5d9b20f7b43c3ffcff1d6a1d6`.
PR/main CI passed 11/11, Package 19/19 with 18 matching consumer reports.
The pilot uses that qualified wheel; five installed feedback cases, 62 module
hashes and startup/owned cleanup passed. Candidate and main Git trees match.
See `.venv/admission-feedback-completed-1015.json`; previous environments/models
and user ignore files remain preserved.

Branch `fix/download-admission-response-validation` addresses a reproduced false
success: real HTTP 200 with `{}` reported an already-active download. Require a
JSON object containing an actual boolean `queued`; never coerce strings/numbers
or treat a missing field as success. Invalid response shape gives no acceptance
claim and asks the user to check download history before retrying. Do not retry
automatically because the server may already have accepted the request.
Twelve malformed shapes and two explicit boolean results pass through the real
client/HTTP seam; existing feedback, manager, sync and startup tests also pass
(63 total). Ruff, 38-file mypy and independent review passed. Check full verify,
PR CI, Package and actual-main CI before completion/pilot refresh. No protocol,
provider, worker, storage or automatic retry behavior changes are in scope.

Earlier checkpoints below are historical.

PR #169 is merged at `45c7f7bfb9e39b6fdb8dc31917c2ac2213074a40`.
PR CI `38800503323` and actual-main CI `38801477126` passed 11/11.
The pilot now uses the PR #168 qualified wheel; its 62 runtime modules match
that main, the wheel and the installed environment. Nine installed lifecycle
cases and final startup/owned cleanup passed. Previous pilot environments and
models remain preserved; `.venv/idle-retirement-completed-1015.json` records proof.

Branch `fix/download-admission-feedback` stays within user-visible download
startup refusal and HTTP 503 admission handling. The real client with an invalid
database path reproduced an uncaught ValueError; readiness failures now return a
safe-upgrade/retry message. A controlled loopback HTTP 503 reproduced raw technical
feedback; report temporary refusal, no queued download, and manual retry guidance.
Do not automatically resubmit. Real client/HTTP tests cover busy queued/running
legacy services without POST/shutdown, rejected admission followed by manual
success, unchanged local download state and non-503 error distinction.
Five new cases plus existing manager/sync/startup cases passed (49 total), Ruff
and 38-file mypy passed. Check full verify, independent review, PR CI, Package
and actual-main CI before claiming completion or refreshing the pilot.
No service protocol, provider, storage, retry policy or startup-threading change
is in scope. Preserve user ignore changes, previous environments and model files.

Earlier checkpoints below are historical.

PR #168 is merged at `f9d40a1a870a02102e1522893e6602d194c57451`.
PR CI passed 11/11 and Package 19/19, with 18 matching consumer reports.
The candidate passed nine installed Windows lifecycle cases. Its source tree
matches that merge; the default pilot still points to PR #167 until main is green.
Actual-main CI `38724926008` passed 10/11: Windows 3.12 failed the cancelled-search
UI test before the replacement result arrived. Do not report that run as green.

Branch `test/cancelled-search-result-synchronization` changes only tests and this
record. A controlled 1.5-second detail response reproduces the same empty-result
assertion with the old fixed 35-pause loop. Reuse the bounded UI condition helper
to require a new search ID and completed search before checking rows and cache.
Cover immediate and delayed responses; keep old-result rejection assertions.
The blocked old response has a 10-second guard to outlast the 5-second completion
bound and is always released in cleanup. Focused four-case tests and Ruff passed.
Check PR CI and actual-main CI before refreshing the pilot. Application source
and build configuration are unchanged by this follow-up. The sdist includes
tests, so its earlier qualification applies to the recorded candidate only;
do not claim the entire later main tree equals that earlier test/doc tree.
User-visible startup-refusal/503 work remains the next separate scope after this
CI gate. Preserve user ignore files, previous environments and model files.

Earlier checkpoints below are historical.

PR #167 is merged at `78268a258562cfc6c729735abf0c2a2892314aed`;
PR/main CI passed 11/11 and Package 19/19 with 18 matching consumer reports.
The installed pilot passed five four-client lifecycle scenarios and uses the
qualified wheel. Remote `bc8483c` adds README text only; it is retained here.

Branch `fix/download-idle-retirement-admission` fixes a reproduced independent
HTTP job admitted after an empty startup snapshot being overtaken by automatic
service replacement. Authenticated `POST /shutdown-if-idle` checks actual active
jobs and fences further admissions under the same per-handler-factory lock as
`POST /jobs`. Busy returns 409 without shutdown; retirement returns 200, rejects
late jobs with 503, and initiates shutdown even if its response write fails.
Clients never fall back to unconditional shutdown or process killing when
conditional retirement is unsupported or uncertain. Explicit stop is unchanged.
The additive endpoint keeps protocol floor 2.1; actual incompatible historical
services without it require an explicit operator restart after downloads finish.
See `docs/hf-download-identity.md` for this compatibility boundary.
Real HTTP regression gates both admission orderings and response loss; the old
path attempted replacement after accepting a new job. Check current PR/CI/Package
and refresh the pilot only from qualified bytes. Preserve all previous model,
environment and user ignore files. No model acquisition or UI change is in scope.
Next separate scope: verify user-visible handling of refused service startup and
503 admission with controlled responses. Participant/network/power-loss acceptance
and prior HF HTTP504 remain separate.

Earlier checkpoints below are historical.

PR #166 is merged at `3c4460a3be2e42d828b8782d9d69912cc912d44c`.
PR/main CI passed 11/11. Four real client processes passed cold startup and
reuse on source and the existing installed pilot; production code was unchanged.

Branch `fix/download-concurrent-service-upgrade` fixes a reproduced legacy-idle
upgrade race. Serialize participating clients with a separate SQLite
`.client-lock` (6-second acquisition limit), then recheck health/version/jobs.
Before spawning, wait up to 3 seconds for actual `.service-lock` ownership to
be released, since HTTP shutdown can precede old worker cleanup. Keep both lock
files and close connections on all paths. Health/reset errors are contained.
Real four-client cases cover cold start, legacy idle replacement, legacy
queued/running snapshots refusing replacement, and delayed old-worker cleanup.
Exactly one legacy retirement is required; bypassing client coordination is
detected. Legacy protocol/version and passive busy states are controlled fixtures,
not an installed historical binary or live-provider workload.
Check current PR/CI/Package; refresh the pilot only from qualified candidate bytes.
Preserve user ignore changes, previous environments and model files. No model
acquisition, schema change or UI work belongs to this scope.
Next separate boundary: a new job submitted after the legacy idle snapshot but
before retirement. Participating `ensure_service_running` callers are serialized;
independent job submission/manual shutdown is not made atomic by this fix.
Participant/network/power-loss acceptance and prior HF HTTP504 remain separate.

Earlier checkpoints below are historical.

PR #165 is merged at `ca5a81ed2de5773cd9877f26cbe602ff937ccbeb`.
PR/main CI passed 11/11; Package passed 19/19 with 18 matching consumer reports.
The qualified installed Windows wheel passed the queued-retry shutdown/restart
scenario. The pilot uses that wheel; previous environments and models remain.

Branch `test/download-concurrent-client-startup` adds a real four-process client
acceptance test: simultaneous cold start against one isolated database/port,
authenticated history reads from every client, exactly one surviving service
process tree, and a second synchronized call reusing the same tree. Terminal
history remains unchanged and controlled shutdown removes all owned descendants.
Windows venv forwarding processes are accounted for using actual client PIDs.
This acceptance found no application defect; production code is unchanged.
Check current PR/main CI before continuing; the installed pilot needs no refresh
for this test-only increment. No model acquisition or provider calls are involved.
Next separate scope: assess concurrent clients facing an incompatible idle
service, using a controlled local fixture. Active-job preservation remains a
required boundary. Participant/network/power-loss and prior HF HTTP504 stay open.

Earlier checkpoints below are historical.

PR #164 is merged at `8e042117f4c6ecdd7d1b6562e4029d2a311f9cac`.
PR and actual-main CI passed 11/11; Package passed 19/19 with 18 matching
consumer reports. The isolated Windows pilot now uses its qualified wheel;
previous environments and model files remain preserved.

Branch `fix/download-shutdown-preserve-retry` fixes the next reproduced boundary:
a queued retry behind cancelled-worker cleanup was wrongly cancelled by shutdown
through the previous process registration. Shutdown now requests cancellation
only when the current stored row is running, using an atomic SQL predicate.
Explicit user cancellation of queued jobs retains its existing behavior.
The real HTTP RED test reopened a cancelled retry; GREEN preserves the same ID,
automatically runs it after restart and completes exactly the second invocation.
Check current PR/CI/Package before merging and refresh the pilot only from the
qualified candidate. No model acquisition or schema change is in this scope.
Next separate scope: assess simultaneous download-service client startup with
controlled local processes; do not broaden to provider networking or new UI.
Participant/network/power-loss acceptance and prior HF live HTTP504 stay open.

Earlier checkpoints below are historical.

PR #163 is merged at `b5a24a41112d6e475e823746a21fc28cda81f217`; PR and
actual-main CI passed 11/11 on their first attempts. Concurrent active duplicate
submissions are covered through real HTTP and controlled local child processes.

Branch `fix/download-cancel-retry-serialization` fixes a reproduced same-target
retry race: a visible cancelled record could be retried before the previous
future finished clearing its process. Track future-to-target ownership and
exclude unfinished targets during atomic claims, recomputing exclusions each
poll so completed futures do not stall retries. Other targets remain eligible.
The real-service RED test observed a second child overtaking old cleanup; GREEN
keeps the retry queued while another target runs, then verifies tracked restart,
same job ID and completion after cleanup. Claim tests cover quoted/multiple
exclusions and unchanged default behavior. Check current CI/Package and refresh
the isolated pilot only from qualified candidate bytes; preserve previous models,
trial roots and user ignore changes. No model acquisition is part of this fix.
Network/participant acceptance and prior HF live HTTP504 remain separate.

Earlier checkpoints below are historical.

PR #162 is merged at `47993fae64167abdd652b08449be63627e6f73d8`; PR and
actual-main CI passed 11/11 on their first attempts. Early-results TUI test
synchronization is complete; application and installed pilot are unchanged.

Branch `test/download-concurrent-duplicates` adds a real-service HTTP regression:
eight mixed-case duplicate submissions while queued, then eight while running
with a free worker slot. Exactly one submission queues the target, all responses
retain one job ID, and the controlled external child runs once before completion.
Two isolated child-only mutation checks detect missing queued/running guards;
they are not existing application defects and change no repository application
files. The related queue/store suite passed; no model was acquired. Check this
branch's current PR/CI before continuing. Preserve user ignore edits, pilot and
all trial files. Next separate scope: same-target cancellation followed by prompt
explicit retry with controlled local processes. HF live HTTP504, concurrent
runtime clients and participant/network/power-loss acceptance remain separate.

Earlier checkpoints below are historical.

PR #161 is merged at `8f946f9e6a42a169c8a2a3f1047986c735fc2f14`.
PR CI passed 11/11; actual-main run 38586122191 passed 11/11 on attempt 2.
Attempt 1 failed the Windows 3.12 early-results TUI test; rerun alone did not
establish a permanent fix, so that test took priority over concurrency review.

Branch `test/ollama-preview-synchronization` reproduces the same empty-prefix
assertion with a controlled 500-ms fast-detail response. The original fixed
15-pause wait failed; bounded waits for the visible prefix and completed search
pass both immediate/delayed responses while preserving all existing assertions.
Application code and installed pilot are unchanged; no model was acquired.
Check this branch's current PR/CI before continuing. Preserve user ignore edits
and trial files. Next separate scope remains concurrent duplicate managed
submissions with a controlled local worker; network/participant gates stay separate.

Earlier checkpoints below are historical.

PR #160 is merged at `5c9d338ca48cd6b47dfcc23f35458478f9218060`; PR and
actual-main CI passed 11/11. It archives the bounded daemon cancellation evidence.

The next installed Windows Ollama runtime-loss acceptance passed on the same
wheel, fixed model and budget. Kill of the owned daemon at 1% progress caused
`failed`, nonzero exit/diagnostic and no cancellation flag while the download
service stayed healthy. Restoring only the daemon preserved that failed record;
explicit same-ID retry completed verified manifest/all five blobs. See
[evidence and limits](../docs/ollama-live-acceptance.md#installed-runtime-loss-during-pull-2026-10-14).
Application code is unchanged. Check this documentation PR/CI before continuing.
Preserve user ignore edits, pilot and all trial/model files. Next narrow review:
concurrent duplicate managed submissions, using a controlled local worker first
without another model acquisition. Concurrent runtime clients, network/power
failure and participant acceptance remain separate. The previous scheduled
HF HTTP504 finding was not rechecked in this runtime-loss trial.

Earlier checkpoints below are historical.

PR #159 is merged at `8be863aafb6f91f2c2e370690cc885657a43c7b3`; PR and
actual-main CI passed 11/11. Its changes only archive installed Ollama recovery.

The next daemon cancellation observation passed with the same installed wheel,
fixed model and budget. Before cancel, seven actual owned-daemon external HTTPS
connections were observed. After managed cancel and two seconds settling, all
50 samples over five seconds had zero active connections while the daemon
remained healthy, no model completed and the job stayed cancelled. Reopen/retry
and manifest/five-blob identities passed. See [evidence and limits](../docs/ollama-live-acceptance.md#installed-daemon-cancellation-observation-2026-10-14).
The first proxy-only nominal pass was inconclusive because redirected blob
traffic bypassed it; the corrected helper inspects actual owned-daemon sockets.
No application changes. Check the documentation PR/CI before continuing; preserve
pilot, user ignore edits and all trial files. Next separate scope: failed Ollama
runtime during a managed pull and explicit retry after runtime restoration.
Concurrent pulls, participant outcomes and network/power failure remain separate.
Prior scheduled live run 38534034347 failed HF search HTTP 504; that finding was
not rechecked or repaired as part of this Ollama observation.

Earlier checkpoints below are historical.

PR #158 is merged at `e2932534483feceb81530eca3915e2a4e642f45f`; PR and
actual-main CI passed 11/11. Its changes archive HF connection interruption.

Installed-candidate Windows Ollama acquisition/reopen passed on the same PR #156
wheel and previously authorized `tinyllama:1.1b` identity/budgets. Managed cancel
at CLI-reported 1% stopped the pull CLI; service/runtime reopen retained the
cancelled job, same-ID retry completed, and manifest/all five blobs verified.
See [evidence and limits](../docs/ollama-live-acceptance.md#installed-current-candidate-2026-10-14).
Only the task-owned executable observation helper changed after an inconclusive
first gate; application code is unchanged. Check the documentation PR/CI before
continuing. Preserve user ignore changes, pilot and all trial/model files.
Next narrow scope: observe Ollama daemon transfer termination after managed
cancel while keeping the daemon alive; current CLI-exit evidence does not prove
that latency. Participant recovery and Ollama network-failure acceptance remain
separate. Do not expand into inference or model deletion.

Earlier checkpoints below are historical.

PR #157 is merged at `6e6832b6631f6aa01e3c5817df910a31ad1c5db5`; PR and
actual-main CI passed 11/11. Its changes only archive interruption acceptance.

The next installed Windows HF network-interruption trial passed on the same
PR #156 wheel. An owned CONNECT relay closed the transfer after a 10 MiB partial
file appeared; the service remained healthy and persisted a failed job.
Restart retained the plan/status; explicit retry reused the job and verified
final size/SHA-256. See [evidence and limits](../docs/hf-recovery-acceptance.md#installed-network-interruption-2026-10-14).
Application code was unchanged. Check the documentation branch/PR CI before
continuing. Preserve user ignore changes, the installed pilot and trial files.
Ollama download acceptance is the next separate stabilization scope; participant,
DNS/timeout, host-outage and SDK automatic-reconnection proof remain unassessed.

Earlier verified runtime checkpoints below are historical.

PR #156 is merged at `3ca8ad0db805ac0c15c9b20d7297e4ce880aa6a2`; actual-main
CI passed 11/11, Package passed 19/19 and all 18 consumer reports matched the
candidate. Queue cancellation is complete and the isolated pilot was refreshed.

The next Windows installed-candidate HF acceptance passed early cancellation,
controlled partial-transfer shutdown and forced service-process stop/restart
with the same pinned 19 MB public sample. Each explicit retry verified bytes and
digest; see [current evidence and limits](../docs/hf-recovery-acceptance.md#installed-current-candidate-2026-10-14).
Application code was unchanged; only a task-owned Windows launcher observation
assumption was corrected. Evidence is bound to the PR #156 wheel, not future
package builds. Preserve user ignore changes and trial files. Network failure,
Ollama and participant acquisition/recovery remain separate acceptance scopes.

Older continuation checkpoints below are historical.

Current verified baseline: PR #155 merged at
`d00e615b9244f2466e4dad5b4ed43515622d18e1`; actual-main CI passed 11/11,
Package passed 19/19 and all 18 consumer reports matched the candidate bytes.
Legacy normalization is complete; the isolated Windows pilot was refreshed.

Branch `fix/download-queued-cancellation` addresses only saturated queues:
claim a job only when a worker slot is available, keep later jobs queued for
immediate cancellation, reuse a completed slot and preserve unclaimed jobs at
shutdown. A real-service regression reproduced premature running status; the
fix and existing worker tests passed. No live model was acquired.
Check this branch's current PR/CI before continuing. Keep real-provider
acquisition/recovery acceptance separate; preserve current user ignore changes.

Previous checkpoints below are historical.

Verified baseline: PR #154 merged at
`429858de027cb6f573592e27b9c954656a0ab900`; actual-main
[CI 38361010107](https://github.com/kefyusuf/llm-terminal/actions/runs/38361010107)
passed 11/11. Startup ownership and isolated development smoke are complete.

The next narrow branch is `fix/download-legacy-id-normalization`: a regression
reproduced UNIQUE failure when a mixed-case record is newer than its canonical
duplicate. Normalize within one immediate transaction, delete losers before
updating retained IDs, and preserve the existing timestamp/row-ID winner policy.
Focused tests verify full visible record preservation, idempotence and rollback
on an injected database update failure. Check the current branch PR/CI before
continuing. Saturated-queue cancellation and live acquisition remain separate.

Earlier checkpoints below are historical.

Current verified main baseline is now PR #153 at
`34e1bbe0cea06247adaac7cab578e522ffdd511e`: actual-main
[CI 38354647054](https://github.com/kefyusuf/llm-terminal/actions/runs/38354647054)
passed 11/11; PR Package passed 19/19 and all 18 consumer reports matched the
candidate bytes. The Windows participant reported that the updated search was
better. The isolated launcher points to the qualified PR merge-source wheel,
whose tracked tree equals actual main; publication remains separate.

The user selected download-service startup ownership as the next narrow fix.
Branch `fix/download-service-startup-ownership` moves initialization/recovery and
worker start after database ownership and socket bind, including smoke startup.
Process regressions reproduced losing-startup mutation and duplicate ownership;
thread-start failure cleanup was independently reviewed and corrected.
Check the branch PR/CI before continuing. Keep legacy normalization collision,
saturated queue cancellation and live download acceptance as separate follow-ups;
do not fold them into this startup correction.

The earlier search checkpoint below is historical.

Verified baseline: PR #152 merged at
`3b82383af248e46fff45d19e8d8b3d58ed71dd4c`; actual-main
[CI 38249673195](https://github.com/kefyusuf/llm-terminal/actions/runs/38249673195)
passed 11/11. The Windows participant reported that search felt somewhat better,
without a measured duration or competitor identity.

Branch `fix/ollama-early-search-results` continues only the Ollama search wait:
fully scored early rows, stable selection, obsolete-work rejection and a
single-attempt optional installed-model probe. Remote retry defaults and final
scoring remain unchanged. See [behavior and evidence](../docs/ollama-search-performance.md#early-visible-results-and-optional-local-probe-2026-10-13).
Check the current PR and CI before continuing; do not infer merge or package
qualification from the local observation. After green checks, merge is authorized
by the user's existing instruction and the isolated Windows pilot may be refreshed.
Participant retest, publication and release-version selection are separate gates.

The records below are historical checkpoints.

## Current verified main checkpoint

[PR #148](https://github.com/kefyusuf/llm-terminal/pull/148) merged the bounded
calibration preload/failure-stage correction as
`0afc411bbd6258195a0248b86f33995e42bbe891`. Its tracked tree matches independently
reviewed head `d19ed3e7e6d7b54ad78da098cb3a214756ef7518`. Both
[PR CI](https://github.com/kefyusuf/llm-terminal/actions/runs/37693435961) and
[actual-main CI](https://github.com/kefyusuf/llm-terminal/actions/runs/37694030839)
passed all 11 jobs. The full local collection had 911 tests: 910 passed and one
POSIX-only test was skipped on Windows. Imports/Ruff, staged 39-file mypy and
CLI/TUI/API/service smoke passed. The first full-test attempt could not remove
an existing temporary root; the successful repeat used a fresh project-owned
root without changing ACLs.

Two fresh owned Ollama 0.35.0 servers completed CUDA calibration with the already
verified `tinyllama:1.1b` files and unchanged 30-second request/120-second worker
limits. The [archived report](../docs/evidence/ollama-cuda-calibration-windows-2026-10-08.json)
binds the tested calibration and scoring hashes; the historical timeout cause
remains unresolved. Owned servers/descendants stopped; model files remain under
ignored `.venv/`. No Docker resources or worktrees were created.

Package CI was not triggered by these calibration-tool/test/document paths.
Historical Package qualification below does not qualify new release bytes.
Version remains 1.0.1; no publication or participant outcome is claimed.

Next, use the [pilot acceptance form](../docs/pilot-acceptance.md) after selecting
an authorized immutable candidate and its installation reports. Collect actual
participant outcomes; do not replace them with maintainer tests. New release
version, publisher/protected environment and promotion remain explicit decisions.
Keep broader calibration separate and preserve current scoring.

The next checkpoint completed on actual main
`88fed3f02bf8607407fbbe4250106c52971147cb`: its
[CI passed 11/11](https://github.com/kefyusuf/llm-terminal/actions/runs/37722565654)
and manually dispatched [Package passed 19/19](https://github.com/kefyusuf/llm-terminal/actions/runs/37736783425).
The downloaded manifest/distribution bytes and all 18 consumer reports were
cross-checked; see [exact-main package inputs](../docs/build-once-candidates.md#2026-10-08-exact-main-candidate-checkpoint)
and [archived summary](../docs/evidence/qualified-main-package-2026-10-08.json).
This supplies immutable development pilot inputs without selecting a new release
version, publishing, messaging participants or claiming real pilot/rollback
outcomes. Those gates remain open.

The 2026-10-08 [installed-worker restore report](../docs/evidence/qualified-pilot-candidate-restore-windows-2026-10-08.json)
then qualified the exact `88fed3f` development wheel against previous `eabf63e5`:
both installed origins/receipts and dependency checks passed; both workers
completed the same pinned 19 MB selected file and shut down cleanly. The previous
worker used a copy of the one-terminal-job backup, rebound its plan to a separate
root, and preserved immutable backups and original files. This closes only that
byte pair's isolated terminal-job/file restore; actual participant tasks and a
new release version's rollback remain open. See [procedure and limits](../docs/qualified-candidate-restore.md#2026-10-08-development-pilot-input-pair).

## Historical source and delivery state

Main delivery: [integration PR #143](https://github.com/kefyusuf/llm-terminal/pull/143)
squash-merged the researched roadmap and implementation stack on 2026-10-05
with user authorization. Actual merge source is
`93395de47aaa0817d21b9b736a075f75cafeef5e`. Its complete tracked file tree
matches reviewed candidate `ca86b0daf912306183bec30ee819560d133cad28`.
Original PRs #118–#142 are closed as superseded by #143, not individually merged;
their descriptions, branches and review/test history are retained.

The candidate's full local collection contained 903 tests: 902 passed and one
POSIX-only venv test was skipped on Windows; that regression passed on Linux
and macOS CI. Imports/Ruff, the staged 39-file type gate and main CLI/TUI/API/service
smoke passed. Independent review found and confirmed fixes for claimed-job
cancel/delete races and POSIX venv interpreter resolution before merge.

Actual main source passed
[CI, 11 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37236972922)
and [Package, 19 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37236975704),
including one candidate producer and 18 consumers of the same distributions.
[Candidate live HF CI, 11 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37236560256)
also passed on Linux/Windows with pinned cancellation/retry and size/digest proof.
This follow-up records those inputs without changing runtime behavior. Read its
current Git/PR commit and applicable verification separately. Requalify changed
runtime, version or release artifacts rather than treating old evidence as current.
No model inference, participant pilot, registry publication or live-release
readiness is claimed; external gates remain below.

The final full local run exposed a calibration timing bug on native
Windows Python 3.12.14: fast responses can share a 15.625 ms monotonic tick.
The merged correction uses a high-resolution elapsed counter while retaining deadline
clocks and all validation. Its deterministic regression failed before the fix;
the 23-test focused suite, historical full 898-test run and exact-source CI/package gates
passed after it. These are simulated protocol/fixture measurements, not a
genuine inference corpus.

The original implementation PRs are: #119 provider discovery, #120 hardware
startup, #121 startup measurement, #122 doctor, #123 score provenance, #124
isolated package verification, #125 revision pinning, #126 path preflight, #127
artifact metadata, #128 download plans, #129 process/partial recovery, #130 owned
removal and service shutdown, #131 build-once candidate, #132 support/release
evidence, #133 versioned exports, #134 measured table updates, #135 HF detail
request contract, #136 child lifetime, #137 SQLite data compatibility, #138
declared memory scenarios, #139 bounded calibration, #140 opt-in live HF CI,
#141 qualified candidate restore/documentation and #142 Windows elapsed-clock
correction. These are historical scope references; their combined delivery is #143.
Pre-integration main was freshly read as `32fe324b97294d32b73795b06a12158e85abe283`.
The independent roadmap proposal #118 is incorporated on main;
the historical mainline roadmap must not replace that research-informed scope.
Read main and the integration PR's actual merge commit from GitHub on continuation.

Preserve the pre-existing untracked `ai_model_explorer.egg-info/`; do not add or
delete it. Build future local candidates from fresh exported source to avoid
changing source-tree generated metadata. Runtime test data and reports live
under ignored `.venv/`. Tests/imports must explicitly override cache DB, download
DB and HF model root into project-owned directories; otherwise Windows imports
can attempt protected AppData writes. Keep TLS verification enabled; a prepared
process-only Windows trust bundle is available in local runtime probe data.

## Completed acquisition evidence

The Windows trial used HF SDK 0.36.2 and Python 3.12.14 with a 19,077,344-byte
GGUF at full commit `499bc8821c6b12b4e53c5bffcb21ec206f212d81` and verified its
SHA-256. It observed 10 MiB partial output before cancellation, reopened the
database and retried successfully. A separate real authenticated service trial
proved active controlled shutdown, process exit, restart, plan preservation,
completion and selected-file removal with sibling preservation. See the linked
JSON reports in `docs/evidence/`. This is not inference, usage permission or
forced-host-crash proof. SDK auxiliary/partial caches are retained by deletion.

The later forced-service-stop trial observed the owned HF child exit and a
successful retry. Exact-source Linux/Windows CI at `6bab5a6` independently
observed 10 MiB partial cancellation and final byte/digest verification.
Installed build-once candidates `6bab5a6` and `5d65b5f` completed a terminal-job
SQLite backup and selected-file restore rehearsal in separate roots, including
previous-worker execution after server-side plan rebinding. See
`docs/qualified-candidate-restore.md`; these are bounded operation proofs.

## Next work and external gates

1. M2 exports, saved comparisons, declared memory scenarios and bounded runtime
   calibration tooling are implemented. Table evidence covers 100/1000-row
   structural versus download-only costs and avoids unchanged cell writes.
   Genuine calibration requires a dedicated prepared local Ollama runtime and
   real repeated inference samples. Preserve numerical scoring until measured
   acceptance supports a formula change. See `docs/runtime-calibration.md`.
2. Qualify prepared Ollama acquisition/recovery with isolated stores and explicit
   size/time limits. HF Linux/Windows and forced-service child-lifetime evidence
   are now recorded. The refreshed 2026-10-05 host probe found an installed
   Ollama client 0.35.0, but the default loopback API was unavailable and the
   local model store contained zero manifests. The earlier 1.47 GB Windows x64
   portable asset metadata is not installation or inference evidence.
   A subsequent owned-server preflight on source `5906e79` passed: version 0.35.0,
   a separate loopback port, cloud explicitly disabled through `/api/status`,
   zero installed models, and owned-process cleanup. The server created only
   empty `blobs`/`manifests` directories. See the archived
   `docs/evidence/ollama-runtime-preflight-windows-2026-10-05.json`.
   On 2026-10-07, the user authorized `tinyllama:1.1b` with at most 1 GB model
   files and 15 minutes. Main `bc50cf1e` passed CI 11/11; its managed
   acquisition/cancel/reopen/retry passed with exact manifest/blob verification.
   Default CUDA calibration failed at the 30-second generation deadline.
   Forced-CPU fallback measured three post-warmup samples at median 81.05 tok/s,
   using the same files. See `docs/ollama-live-acceptance.md`; this named Windows
   path alone does not qualify CUDA, other platforms or a broad calibration corpus.
   On 2026-10-08, unchanged CUDA calibration passed twice. A subsequent bounded
   preload separation passed two fresh-server GPU runs at median 337.86/334.51
   tok/s, with three 64-token samples each and unchanged 30/120-second budgets.
   See the script-hash-bound GPU report in the acceptance record. The original
   transient timeout cause remains unresolved; broader qualification remains open.
   Owned processes/descendants were cleaned up; model files remain for review.
3. SQLite data compatibility and isolated previous-worker/file restore passed
   for the named candidates. Actual main `eabf63e5` subsequently passed Package
   19/19 and the installed-worker restore rehearsal against previous `6bab5a6`:
   one terminal job, isolated roots, rebound plan, pinned file size/digest,
   immutable backup preservation and controlled shutdown. See
   `docs/evidence/qualified-main-restore-windows-2026-10-05.json` and
   `docs/qualified-candidate-restore.md`. This does not qualify a future changed
   release version or production rollback; additive migrations alone are not
   downgrade proof.
4. Keep TestPyPI/PyPI account/name, publisher/protected environment, chosen new
   version, attestations, genuine participant pilot and actual promotion as
   explicit gates. Do not fabricate human outcomes, send unapproved messages or
   publish/merge merely because earlier CI was green.

Private GitHub vulnerability reporting was enabled and verified during the
support documentation scope. Refresh repository state before relying on it.
The support matrix and release checklist are evidence records, not instructions
from past logs. No release or production readiness claim is warranted yet.
