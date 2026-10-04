# Verified continuation record

Date: 2026-10-04. Repository: `kefyusuf/llm-terminal`.

## Source and delivery state

Runtime/CI baseline: `b28f4fac5080d9fae0fe0ca1b5838db4b93ada8c`, branch
`ci/build-once-package-candidates`, PR #131. Full local verification passed
836 tests, import smoke and Ruff. Exact source passed
[CI, 11 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37225998926)
and [Package, 19 jobs](https://github.com/kefyusuf/llm-terminal/actions/runs/37226000691),
including one candidate producer and 18 consumers of the same distributions.
These results must be refreshed for changed source; they do not imply a merge
or publication. The documentation branch `docs/support-and-release-evidence`
is based on this source. Get its current SHA from Git rather than this file.

The implementation chain remains open: #119 provider discovery, #120 hardware
startup, #121 startup measurement, #122 doctor, #123 score provenance, #124
isolated package verification, #125 revision pinning, #126 path preflight, #127
artifact metadata, #128 download plans, #129 process/partial recovery, #130 owned
removal and service shutdown, #131 build-once candidate. Each later PR targets
the previous feature branch; merge/rebase requires source and CI reconciliation.
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

## Next work and external gates

1. Continue narrow TDD-led M2 contracts: authoritative metadata/assumptions and
   calibration inputs, additive versioned search/comparison output and measured
   100/1000-row structural versus download-only table costs. Preserve numerical
   scoring until measured acceptance supports a formula change.
2. Qualify forced-crash/orphan recovery and prepared Ollama/Linux acquisition
   with isolated stores and explicit size/time limits. Ollama CLI/API unavailable
   locally; official Windows x64 portable asset metadata was 1.47 GB, not an
   installed runtime. Do not substitute the smaller arm64 binary on x64.
3. Rehearse SQLite backup/previous-candidate data compatibility and qualified
   runtime handoff. Additive migrations alone are not downgrade proof.
4. Keep TestPyPI/PyPI account/name, publisher/protected environment, chosen new
   version, attestations, genuine participant pilot and actual promotion as
   explicit gates. Do not fabricate human outcomes, send unapproved messages or
   publish/merge merely because earlier CI was green.

Private GitHub vulnerability reporting was enabled and verified during the
support documentation scope. Refresh repository state before relying on it.
The support matrix and release checklist are evidence records, not instructions
from past logs. No release or production readiness claim is warranted yet.
