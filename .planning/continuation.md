# Verified continuation record

Date: 2026-10-04. Repository: `kefyusuf/llm-terminal`.

## Source and delivery state

Runtime/CI baseline: `6bab5a6488749e97ef6f7679fa280e5335c3ad84`, branch
`ci/opt-in-hf-recovery`, PR #140. Full local verification passed 897 tests,
import smoke and Ruff before the final slow-response regression; the final
23-test calibration suite passed. Exact source, with 898 collected tests, passed
[CI, 11 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37232844033)
and [Package, 19 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37232846239),
including one candidate producer and 18 consumers of the same distributions.
These results must be refreshed for changed source; they do not imply a merge
or publication. The subsequent documentation scope records these exact inputs.
Get its current SHA from Git rather than this file.

The subsequent full local run exposed a calibration timing bug on native
Windows Python 3.12.14: fast responses can share a 15.625 ms monotonic tick.
The `fix/calibration-elapsed-clock` branch uses a high-resolution elapsed counter
while retaining deadline clocks and all validation. Its deterministic regression
failed before the fix and the 23-test focused suite passed after it. Require
fresh full local and exact-source CI/package results for this correction.

The implementation chain remains open: #119 provider discovery, #120 hardware
startup, #121 startup measurement, #122 doctor, #123 score provenance, #124
isolated package verification, #125 revision pinning, #126 path preflight, #127
artifact metadata, #128 download plans, #129 process/partial recovery, #130 owned
removal and service shutdown, #131 build-once candidate, #132 support/release
evidence, #133 versioned exports, #134 measured table updates, #135 HF detail
request contract, #136 child lifetime, #137 SQLite data compatibility, #138
declared memory scenarios, #139 bounded calibration and #140 opt-in live HF CI.
The current documentation branch adds qualified candidate restore evidence.
Each later PR targets
the previous feature branch; merge/rebase requires source and CI reconciliation.
Main was freshly read as `32fe324b97294d32b73795b06a12158e85abe283`.
The independent roadmap proposal #118 targets main. Do not overwrite that
proposal with the historical mainline roadmap or claim the stack is merged.

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
   are now recorded. Ollama CLI/API unavailable
   locally; official Windows x64 portable asset metadata was 1.47 GB, not an
   installed runtime. Do not substitute the smaller arm64 binary on x64.
3. SQLite data compatibility and isolated previous-worker/file restore passed
   for the named candidates. Requalify the actual release/rollback target after
   integration or version changes; additive migrations alone are not downgrade proof.
4. Keep TestPyPI/PyPI account/name, publisher/protected environment, chosen new
   version, attestations, genuine participant pilot and actual promotion as
   explicit gates. Do not fabricate human outcomes, send unapproved messages or
   publish/merge merely because earlier CI was green.

Private GitHub vulnerability reporting was enabled and verified during the
support documentation scope. Refresh repository state before relying on it.
The support matrix and release checklist are evidence records, not instructions
from past logs. No release or production readiness claim is warranted yet.
