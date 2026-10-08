# Build-once package candidates

Package CI builds one wheel and sdist from its checked-out commit in a dedicated
Ubuntu/Python 3.12 job. Twine checks distribution metadata before upload. A schema
1 manifest records the source commit, distribution/version and both filenames,
byte counts and SHA-256 digests. The workflow preserves the files and manifest
as `package-candidate-<source-sha>` for 30 days, subject to repository retention.

All 18 existing installation lanes download this same candidate. Each verifies
the manifest against its checked-out source and the actual file bytes/metadata
before invoking the isolated installer. Fifteen wheel lanes cover Linux, Windows
and macOS on Python 3.10–3.14; three sdist lanes cover each OS on Python 3.12.
Sdist installation can build a wheel internally, but the tested source archive
is the same input. Reports retain the input artifact hash and resolved dependency
versions. No matrix job regenerates the build-job distributions.

This uses [GitHub workflow artifact sharing](https://docs.github.com/en/actions/tutorials/store-and-share-data).
The manifest equality check fails the consumer instead of relying only on an
artifact transport warning. It is an unsigned provenance record, not a PyPI
attestation or an independent proof that a supplied SHA produced arbitrary local
bytes. In CI, source identity is supplied by `github.sha` and checkout. PR events
can use a synthetic merge commit; use the manifest's actual source revision,
not an assumed PR head. Manual branch runs qualify that branch's exact commit.

For a local trial, export a clean commit to a fresh source directory before
building so user-owned generated metadata and uncommitted files are excluded:

```console
python -m build --outdir <fresh-dist-dir>
python scripts/package_candidate.py --dist-dir <fresh-dist-dir> --source-sha <full-commit> --manifest <fresh-dist-dir>/candidate-manifest.json
python scripts/package_candidate.py --dist-dir <fresh-dist-dir> --source-sha <full-commit> --manifest <fresh-dist-dir>/candidate-manifest.json --verify
python scripts/verify_installed_package.py --dist-dir <fresh-dist-dir> --kind wheel --report <wheel-report.json>
python scripts/verify_installed_package.py --dist-dir <fresh-dist-dir> --kind sdist --report <sdist-report.json>
```

The manifest helper requires `packaging`, supplied by the build tooling. A local
report must identify its actual source separately; passing a SHA argument alone
does not establish clean-source provenance.

## 2026-10-08 exact-main candidate checkpoint

Actual main `88fed3f02bf8607407fbbe4250106c52971147cb` passed
[CI, 11/11](https://github.com/kefyusuf/llm-terminal/actions/runs/37722565654)
and a manually dispatched [Package run, 19/19](https://github.com/kefyusuf/llm-terminal/actions/runs/37736783425).
The [archived qualification summary](evidence/qualified-main-package-2026-10-08.json)
records the manifest and all 18 report identities. Downloaded distribution bytes
were independently checked against the manifest; all consumer reports matched
the expected OS/Python/kind lanes, version and input artifact hashes and reported
fresh environments, checkout exclusion, removed Python path overrides, 24 module
origins and all 11 installation/startup checks.

| Input | Bytes | SHA-256 |
|---|---|---|
| `ai_model_explorer-1.0.1-py3-none-any.whl` | 145574 | `ec49018eb98c52bd4cccdd108671b89f39c7ce286aa12eec685c31516fe3496d` |
| `ai_model_explorer-1.0.1.tar.gz` | 231759 | `76ed67855a31c61443a0a561b21106a14a54d9540c9f292750e50cfd6e487ed8` |

The original candidate artifact is
`package-candidate-88fed3f02bf8607407fbbe4250106c52971147cb` in that Package run;
its configured retention is 30 days. Downloaded bytes and raw consumer reports
are retained under ignored `.venv/package-main-88fed3f-37736783425/`. The summary
retains each raw report's SHA-256, not its temporary runtime paths. Retention is
not permanent archival. Use these exact bytes for any subsequently authorized
development pilot; changing source/version requires requalification. Version
1.0.1 remains a development candidate, not a published release or a completed
participant pilot. A subsequent [isolated terminal-job/file restore](qualified-candidate-restore.md#2026-10-08-development-pilot-input-pair)
qualified these exact wheel bytes against previous `eabf63e5`; it does not qualify
active-job migration or a new release version's production rollback.

## Publication gates

This workflow does not publish, grant OIDC write permission or create releases.
A future approved publishing job must retrieve these qualified inputs, verify
source/hash identity again and upload only the wheel/sdist named in the manifest,
without rebuilding or uploading the JSON file as a distribution.

Before TestPyPI/PyPI rehearsal, verify the account/project name, protected GitHub
environments, review requirements and configured Trusted Publisher. Follow the
[PyPI publisher guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/):
OIDC permission belongs only to the publishing job, which retrieves the preserved
distributions. Capture publish attestations and separately verify a TestPyPI
installation with controlled dependency sourcing. These external configuration,
release authorization and pilot gates remain unverified here. Version 1.0.1
artifacts in development CI are candidates, not an authorized new release.
