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
