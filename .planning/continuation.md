# Verified continuation record

Date: 2026-10-08. Repository: `kefyusuf/llm-terminal`.

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
