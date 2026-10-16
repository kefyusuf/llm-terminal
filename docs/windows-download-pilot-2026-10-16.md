# Windows download pilot: 2026-10-16

Status: **partial pilot**. One participant (`windows-pilot-01`) selected the
small public HF sample, completed its download and confirmed the two reported
UI issues resolved. This records submitted outcomes and read-only artifact
checks. It does not establish participant self-install, recovery, model inference
or stable-release acceptance. No participant name, credentials or local paths
are included.

## Exact retest input

| Field | Value |
|---|---|
| Version | Development `1.0.1`; unpublished |
| Candidate source | `133d35540183f40d55e129d07a6e34c3ab4053f5` |
| Merged application source | `92a532144b17de7018c446a47b313f4cb5f86440` ([PR #173](https://github.com/kefyusuf/llm-terminal/pull/173)) |
| Wheel | `ai_model_explorer-1.0.1-py3-none-any.whl`, 148,899 bytes |
| Wheel SHA-256 | `6f07998c1d3cc109aab2954e97e73ccbac368ed2d00f68e41bfb40d80d1d7541` |
| Candidate manifest SHA-256 | `83b673a4570f94d7598541676b0219e29f74586700377f7ee23dfff359e3d4a2` |
| Participant runtime | Native Windows, AMD64, Python 3.12.14 |
| Destination | Separate Windows pilot models directory; existing models preserved |

The candidate and merge have the same Git tree. The wheel was installed outside
the checkout; all 62 runtime modules match both the wheel and merged source.
[PR CI](https://github.com/kefyusuf/llm-terminal/actions/runs/38840482430) and
[application-main CI](https://github.com/kefyusuf/llm-terminal/actions/runs/38841052273)
passed 11/11. [Package qualification](https://github.com/kefyusuf/llm-terminal/actions/runs/38840482356)
passed 19/19, with 18 consumer reports matching the preserved wheel/sdist hashes.
These are maintainer/CI checks, not participant installation outcomes.

## Agreed artifact and observations

- Repository: `ggml-org/models-moved`.
- File: `tinyllamas/stories15M-q4_0.gguf`.
- Revision: `499bc8821c6b12b4e53c5bffcb21ec206f212d81`.
- Final size: 19,077,344 bytes.
- SHA-256: `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`.
- License metadata: unknown; no model-use permission or inference claim.

The participant chose the approximately 19 MB sample. The installed participant
session recorded one completed job; a subsequent read-only check independently
verified final size and SHA-256 inside its isolated destination. Transfer time,
wire bytes and a throughput target were not measured.

The earlier PR #172 input completed the transfer but left stale active-download
feedback and a clipped table. PR #173 fixed those blockers. Asked whether the
Download column was visible and the active message cleared after completion,
the participant replied: “İkisi de düzeldi” (both issues resolved).

The retest used a fresh, separate history and destination, rather than changing
the earlier participant database or replacing its runtime while active. The
earlier completed artifact remains preserved. This is not an upgrade rehearsal.

| Participant task | Outcome | Evidence or limit |
|---|---|---|
| Self-install outside checkout | Unverified | Maintainer prepared the installed environment |
| Open the prepared selection | Passed for this prepared download flow | Participant completed the selected transfer; search was bypassed by the pilot launcher |
| Understand plan, destination and license | Unverified | Do not infer understanding from successful transfer |
| Complete the agreed selected download | Passed | Completed job and matching final size/SHA-256; no elapsed-time claim |
| See Download and cleared active message | Passed | Explicit participant confirmation after the PR #173 retest |
| Cancel, reopen and retry | Unverified | No participant cancellation/restart/retry observation in this session |
| Remove the managed file | Unverified | No removal requested or performed |
| Upgrade/recover or uninstall | Unverified | No participant observation; separate qualified inputs/procedures required |

Earlier search feedback remains bound to its [earlier input](pilot-acceptance.md).
It does not establish search performance on this candidate. Maintainer tests
and live rehearsals remain separate from the participant rows above.

## Next bounded step

Continue with participant cancellation, reopening and retry of the same agreed
small artifact in isolation. If the transfer finishes too quickly to observe
cancellation, record that task unverified rather than acquiring a larger model
or claiming success. Preserve completed files and unrelated services. Controlled
service restart is not host/power-loss proof. The remaining release and publishing
gates in [release readiness](release-readiness.md) still apply.
