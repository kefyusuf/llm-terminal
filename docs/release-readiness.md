# Release evidence and promotion gates

Current checkpoint (2026-10-08): main
`0afc411bbd6258195a0248b86f33995e42bbe891` passed
[CI, 11/11](https://github.com/kefyusuf/llm-terminal/actions/runs/37694030839)
after [PR #148](https://github.com/kefyusuf/llm-terminal/pull/148) merged.
Calibration changes preserved request/worker limits and scoring; two bounded
native Windows GPU runs are recorded separately from the historical failed trial.
No new Package candidate was scheduled for these paths. The qualification below
is historical artifact evidence, not a newly versioned release.

Reviewed on 2026-10-05. Integration PR #143 incorporates the research roadmap
#118 and implementation stack #119–#142, including the Windows calibration
elapsed-clock correction. It squash-merged as
`93395de47aaa0817d21b9b736a075f75cafeef5e`; actual main CI passed 11/11 and its
Package qualification passed 19/19. Independent review's claimed-cancellation
and POSIX interpreter findings were fixed before merge. Original PRs are closed
as superseded, with history retained. The current
[continuation record](../.planning/continuation.md) separates done and pending work.

| Gate | Current evidence | Status |
|---|---|---|
| M0 startup, diagnosis, estimate explanations, isolated installs | PRs #119–#124; startup reports and installed smoke; heuristic provenance remains explicit | Implemented; historical startup timing is not a new candidate timing sample |
| M1-D1 HF identity/license/path/disk plan | PRs #125–#128; durable pinned identity, metadata, known-byte reservation and explicit unknowns | Implemented and tested; no license permission inferred |
| M1-D2 acquisition acceptance | PRs #129–#130/#136/#140; Windows HF service stop/removal; hosted HF cancel/retry; [Ollama named-model trial](ollama-live-acceptance.md) with controlled reopen/retry and manifest/blob hashes | Bounded named-artifact operations verified; host/power-loss proof pending |
| M1-P1 declared operation support | [Support matrix](support-matrix.md); hosted package reports and native Windows trials | Packaging verified; optional inference/backend combinations experimental |
| M1-O1 catalog source decision | [ADR 0001](adr/0001-ollama-catalog-source.md), parser fixtures and observable shape errors | Decision recorded; no undocumented global API migration |
| M3-L1 single release input | PR #131 exact source passed 11 CI jobs and 19 Package jobs; manifest and one preserved wheel/sdist pair | Candidate qualification implemented; no TestPyPI/PyPI publication |
| M3-L2 security/support | [Private report route](../SECURITY.md) enabled; issue tracker for non-sensitive defects | Route verified; no support SLA or completed pilot claimed |
| M2 selection/calibration/export/table cost | PRs #133–#139; exports/comparisons, memory scenarios, table costs, bounded calibration tool; named-model CPU and repeated CUDA samples in [live record](ollama-live-acceptance.md) | Original transient CUDA timeout cause unresolved; broad comparable corpus pending; numerical scoring unchanged |
| Previous-candidate restore | [Installed worker/file rehearsal](qualified-candidate-restore.md) with historical pairs and development pilot input `88fed3f` against previous `eabf63e5` | Named byte-pair isolated terminal-job/file restore verified; actual participant recovery, a future release version and production rollback remain unqualified |

## Candidate review and publication

Use a clean exact commit and retrieve its candidate manifest plus distribution
files. Confirm all required CI conclusions and each installation report's input
hash. PR merge-event source can differ from a branch head; source identity must
match the artifact being promoted. Changing source/version requires new artifact
qualification. Do not rebuild a passing candidate during publication.

The development package is still version 1.0.1; select an authorized unused
release/prerelease version and qualify that version before publishing. Verify
PyPI/TestPyPI account and name ownership, protected GitHub environment reviewers,
Trusted Publisher and attestations. No token or account value belongs in this
repository. [Build-once candidates](build-once-candidates.md) describes the input
boundary and official publishing references. Main integration is complete;
publication has not occurred and remains subject to the gates above.

## Pilot evidence

A pilot is complete only when real participants supply actual outcomes. Record
candidate source/hash, platform/Python, installation result, model selection,
destination/revision, download/cancel/retry behavior and upgrade/uninstall
outcome. Request only participant-submitted redacted diagnostics such as offline
doctor JSON. Exclude credentials, private model identities and unrelated files.
Do not invent feedback or automatically message participants. Resolve blocking
defects before stable promotion and publish the tested/experimental support
matrix with known limits. No participant outcomes have been collected here.

Use the [pilot acceptance form](pilot-acceptance.md) to bind submitted outcomes
to candidate bytes and to keep failed, blocked and unverified tasks explicit.
Preparing the form does not complete the pilot or authorize invitations.

## Upgrade, backup and withdrawal

Stop the owned service gracefully and confirm terminal active jobs before
changing its package or configuration. Back up the SQLite jobs database with
SQLite's backup mechanism, not an uncontrolled copy of a live database; preserve
configured model files separately. Record the package version, source and file
hashes with the backup. Cache data is not a substitute for model or job ownership.

Upgrades add nullable job metadata/plan columns and retain legacy records. This
does not establish arbitrary downgrade safety: older workers can ignore pinned
identity or newer protocol fields. A previous candidate must be independently
qualified, installed in a fresh environment and exercised with a copied backup
before permitting it to process jobs. Keep old workers away from active/shared
destinations during the rehearsal. No previous-version downgrade rehearsal is
claimed by the additive-column tests.

The [bounded store rehearsal](sqlite-compatibility-rehearsal.md) now demonstrates
terminal-job data-read compatibility between the exact `cf0ca4d` and `5d65b5f`
candidates on Windows. It does not qualify older workers for shared destinations
or establish a production downgrade. Use a fresh backup and independent runtime
qualification for the actual rollback target.

The subsequent [installed-worker rehearsal](qualified-candidate-restore.md)
qualified `5d65b5f` against data written by `6bab5a6`: a terminal-job backup and
selected model file were restored into separate roots, the previous server
recomputed its plan before requeue, and both workers completed with matching
bytes/digest. Both original files and immutable backups retained their hashes.
This extends the read-only evidence without authorizing production rollback or
claiming arbitrary version compatibility.

For a bad release, suspend promotion, document the affected source/artifacts and
prepare a tested hotfix. An approved PyPI yank is index metadata, not an automatic
client rollback; exact pins may still select yanked files. Follow the official
[yanking guidance](https://docs.pypi.org/project-management/yanking/) and notify
users only through an authorized release/support communication. Uninstalling the
Python distribution does not remove user databases or downloaded models; any
data removal is explicit and follows managed ownership rules.

## Maintenance

The repository owner maintains dependency intent/locks and provider fixtures;
no calendar SLA is inferred. Keep daily live checks distinct from deterministic
CI, attach actionable provider/error-code evidence and contain parser drift with
a failing fixture before a fix. Security reports use the verified private route.
Record candidate source, distribution hashes, CI/run links, platform/runtime
scope and remaining gates for every promotion decision.
