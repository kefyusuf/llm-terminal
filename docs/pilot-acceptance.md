# Candidate-bound pilot acceptance

Status: prepared on 2026-10-08; no participant outcomes collected. This form
implements the evidence contract in [release readiness](release-readiness.md).
It is not an invitation, publication authorization or completed pilot report.

A qualified development input is now available: source
`88fed3f02bf8607407fbbe4250106c52971147cb`, version 1.0.1, with
[exact distribution hashes and 18 installation reports](build-once-candidates.md#2026-10-08-exact-main-candidate-checkpoint).
The candidate-manifest SHA-256 is
`b75d54ecd58bf761a310718f22432bf13af2fb9ae7ccd01369090b00f5ead47f`.
Its available bytes can populate the identity fields after a pilot is authorized;
the task outcomes remain unverified. This does not choose a public release version
or supply an independently qualified upgrade/rollback pair for these artifacts.

## Prepare one immutable input

Before a participant starts, the maintainer selects an authorized candidate and
records its full source revision, version, wheel/sdist filenames, byte counts,
SHA-256 hashes and candidate-manifest hash. Link the exact-source CI run and
matching installation reports from [build-once qualification](build-once-candidates.md).
Do not substitute the latest main, a mutable branch, or rebuilt distributions.
Current main CI does not supply new package bytes or authorize version 1.0.1
for publication. Leave unavailable evidence blocked rather than using old hashes.

Provide the candidate's own tested installation instructions and known support
limits. Agree with the participant on provider, exact public model artifact,
destination, disk and time budget before acquisition. A model license must be
reviewed separately; the application license does not grant model permission.
Keep their existing models and databases intact. Use the candidate's documented
owned-service shutdown, backup and removal procedures for recovery tasks.

## Participant record template

Copy one record per consenting participant. Use an opaque participant ID, not
their name, contact details, home-directory paths or credentials. Do not request
raw provider logs, private model identities or unrelated files. Optional
`ai-model-explorer-cli doctor --offline --json` output must be reviewed by the
participant before submission; follow [doctor privacy limits](doctor.md).

```text
Participant ID: pending
Consent to submit this record: pending
Candidate full source revision/version: pending
Candidate manifest SHA-256: pending
Installed distribution filename/bytes/SHA-256: pending
Exact-source CI and matching installation-report links: pending
OS/architecture/Python/provider runtime version: pending
Agreed public artifact/revision or digest: pending
Agreed disk/time budget and isolated destination label: pending
Task outcomes and participant-submitted evidence: pending
Blocking issue links and retest candidate identity: pending
Participant assessment of selection/usefulness: pending
```

Each task below needs its own outcome: `passed`, `failed`, `blocked`, or
`unverified`, with the observed behavior and evidence. Blank fields and tasks not
attempted are unverified. A maintainer rehearsal cannot count as a participant
outcome. Do not copy successful CI or GPU measurements into this task table.

| Task | Required observation | Outcome | Evidence / issue |
|---|---|---|---|
| Install outside the source checkout | Candidate identity and installed entrypoints work on the recorded platform | Unverified | Pending |
| Start and select a model | Search/input remain usable; participant understands estimated scores and unknown facts | Unverified | Pending |
| Review the download plan | Exact artifact/revision, destination label, known bytes and license/source agree with the selection | Unverified | Pending |
| Download within the agreed budget | Final selected artifact identity/size/digest agree where available | Unverified | Pending |
| Cancel, reopen and retry | Job state survives controlled owned-service restart; retry completes without changing the selection | Unverified | Pending |
| Remove the selected managed file | Selected file is removed; unrelated files/models remain intact | Unverified | Pending |
| Upgrade and recover | Only an explicitly selected, independently qualified pair; backup preserved and copied state used in isolation | Unverified | Pending |
| Uninstall the application | Documented application removal; databases/models are preserved unless their removal was separately requested | Unverified | Pending |

If acquisition or cancellation cannot be observed within the agreed limits,
record it blocked/unverified and stop that task; do not increase limits or acquire
another model without agreement. Controlled restart is not host/power-loss proof.
Do not deliberately interrupt unrelated services or test rollback against shared
active destinations. An absent qualified upgrade pair leaves that task blocked.

## Review the pilot

The maintainer records participants attempted/completed, outcomes by task and
platform, unresolved blockers and retest evidence. Rebind every retest to its
actual candidate; changing a version or bytes invalidates the previous input
identity. Keep participants' submitted usefulness feedback separate from
maintainer interpretation. A partial pilot remains partial; do not invent a
completion rate, sample-size threshold or successful platform coverage.

Stable promotion requires genuine pilot blockers resolved plus the remaining
release gates, including qualified release/rollback inputs and explicit promotion
authorization. Broader throughput accuracy and other backend/model combinations
remain separate acceptance work.
