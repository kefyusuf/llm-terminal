# Release evidence and promotion gates

Reviewed on 2026-10-04 against the unmerged implementation stack through
`b28f4fac5080d9fae0fe0ca1b5838db4b93ada8c` (PR #131). Main is not implied to
contain this stack. The release roadmap proposal is PR #118; the current
[continuation record](../.planning/continuation.md) separates done and pending work.

| Gate | Current evidence | Status |
|---|---|---|
| M0 startup, diagnosis, estimate explanations, isolated installs | PRs #119–#124; startup reports and installed smoke; heuristic provenance remains explicit | Implemented; historical startup timing is not a new candidate timing sample |
| M1-D1 HF identity/license/path/disk plan | PRs #125–#128; durable pinned identity, metadata, known-byte reservation and explicit unknowns | Implemented and tested; no license permission inferred |
| M1-D2 Windows HF acceptance | PRs #129–#130; real partial cancellation/retry, controlled service restart, bytes/digest and selected-file removal | Verified bounded Windows operation; forced crash/Ollama/Linux acquisition still pending |
| M1-P1 declared operation support | [Support matrix](support-matrix.md); hosted package reports and native Windows trials | Packaging verified; optional inference/backend combinations experimental |
| M1-O1 catalog source decision | [ADR 0001](adr/0001-ollama-catalog-source.md), parser fixtures and observable shape errors | Decision recorded; no undocumented global API migration |
| M3-L1 single release input | PR #131 exact source passed 11 CI jobs and 19 Package jobs; manifest and one preserved wheel/sdist pair | Candidate qualification implemented; no TestPyPI/PyPI publication |
| M3-L2 security/support | [Private report route](../SECURITY.md) enabled; issue tracker for non-sensitive defects | Route verified; no support SLA or completed pilot claimed |
| M2 selection/calibration/export/table cost | Existing heuristic provenance and HF plan JSON are partial foundations | Remaining development/evidence; not represented as complete |

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
boundary and official publishing references. Publication and merge are separate
actions from preparing PRs; neither has happened in this work.

## Pilot evidence

A pilot is complete only when real participants supply actual outcomes. Record
candidate source/hash, platform/Python, installation result, model selection,
destination/revision, download/cancel/retry behavior and upgrade/uninstall
outcome. Request only participant-submitted redacted diagnostics such as offline
doctor JSON. Exclude credentials, private model identities and unrelated files.
Do not invent feedback or automatically message participants. Resolve blocking
defects before stable promotion and publish the tested/experimental support
matrix with known limits. No participant outcomes have been collected here.

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
