# Installation outside the checkout

The Package workflow verifies built artifacts in fresh virtual environments whose working directories are outside the source tree. Earlier checks installed a wheel but imported modules from the checkout working directory; that could mask missing package files.

## Verification boundary

`scripts/verify_installed_package.py` is a standard-library developer driver. It selects exactly one wheel or sdist from the supplied distribution directory, creates an owned temporary environment and empty working directory outside the checkout, and installs the selected artifact with its declared dependencies. Sdist installation performs its own isolated source build. No development requirements or editable installation are used.

Application/token/runtime configuration and Python path overrides are removed from the child environment. Model and database paths point into the owned temporary directory. Pip's persistent user configuration and target/user/prefix overrides are disabled; transport settings such as an explicitly configured index, proxy or certificate remain in effect. Dependency resolution is from package metadata at run time rather than the committed development locks.

The verifier checks:

1. `pip check` for dependency consistency.
2. Required entry modules and each application package import from within the new environment and are listed in the installed distribution's files, with no checkout origins. The import probe uses Python isolated mode (`-I`).
3. Both installed console scripts: CLI version/offline doctor, schema-1 hardware-plan/saved-comparison JSON and a headless TUI mount/exit.
4. The installed REST health endpoint on an ephemeral loopback port.
5. The installed download service's health and authenticated jobs endpoints, with isolated empty state and no model downloads.
6. The real installed download-service client launcher. A recording wrapper delegates to the actual `Popen`, asserts the installed interpreter path, waits for the bounded service smoke subprocess and cleans up its owned handle. Windows exercises the existing `pythonw.exe` launch path; other platforms exercise the existing detached Python launch path.

The original source directory can remain on disk for the driver and build artifacts. It is not a working directory, Python path, or module origin for the installed application checks. No shared runtimes/services are started or stopped. Smoke mode avoids remote model searches and inference. This is installation/runtime-startup evidence, not a model download, GPU inference, live-provider or deployment certification.

## CI matrix and evidence

The Package matrix runs wheel installation on Linux, Windows and macOS for all Python minor versions claimed by `requires-python`: **3.10, 3.11, 3.12, 3.13 and 3.14**. Separate sdist jobs run on each operating system with Python 3.12. Thus source-install evidence covers those three OS lanes, not every OS/Python cross-product for sdists.

Each job uploads `package-report.json`, including status, artifact name/SHA-256, Python/platform, package version, installed module origins relative to the environment, resolved dependency versions and completed checks. Reports contain no machine-specific temporary paths or token values. A failed rerun replaces an old success with a failed report; an interrupted run remains incomplete. Logs and job status remain necessary for failure diagnosis. Build/setup failures before the verifier starts can have no report.

The current workflow now [builds one candidate](build-once-candidates.md) before
the matrix and verifies the same wheel/sdist inputs in every consumer. The extra
candidate job makes 19 workflow jobs while retaining 18 installation lanes.

The native Windows Python **3.12.14** wheel and sdist checks passed for package version **1.0.1** on 2026-10-04. Their exact artifacts and resolved dependencies are recorded in [wheel evidence](evidence/package-windows-wheel-2026-10-04.json) and [sdist evidence](evidence/package-windows-sdist-2026-10-04.json). These are pre-commit local candidate reports, not a release or evidence for other machines. CI uploads fresh reports for its own artifacts and commit.

## Running locally

Use a clean output directory to avoid ambiguous/stale candidates:

```console
python -m build --outdir .venv/package-candidate
python scripts/verify_installed_package.py --dist-dir .venv/package-candidate --kind wheel --report .venv/wheel-report.json
python scripts/verify_installed_package.py --dist-dir .venv/package-candidate --kind sdist --report .venv/sdist-report.json
```

Run with the Python minor version being evaluated. The driver needs no test dependencies, but the build command needs `build`, and install/build dependency resolution needs the configured package index. Keep TLS validation enabled; use an approved trust bundle for a private CA. Do not set the temporary-directory location inside the checkout: the verifier rejects that configuration. Each process step has a timeout; CI also has a job timeout. Temporary environments are removed after verification, while reports remain at the specified paths.

This implements the R3 installation gate proposed in PR #118, following R2 in PRs #122 and #123. It does not publish artifacts, change package support metadata, merge branches or deploy anything.
