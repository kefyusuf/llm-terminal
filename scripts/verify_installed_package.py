"""Verify a wheel/sdist in a fresh environment outside the checkout (stdlib driver)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

# Execute the same validator in the driver tests and isolated installed interpreter.
ORIGIN_VALIDATOR = """
import sys
from pathlib import Path

def assert_installed_origin(origin, prefix, checkout, recorded):
    origin, prefix, checkout = origin.resolve(), prefix.resolve(), checkout.resolve()
    recorded = {path.resolve() for path in recorded}
    if not origin.is_relative_to(prefix) or origin.is_relative_to(checkout) or origin not in recorded:
        raise ValueError("module does not belong to the installed distribution: " + str(origin))
"""
exec(ORIGIN_VALIDATOR)

LAUNCHER_VALIDATOR = """
from pathlib import Path

def assert_launcher_interpreter(executable, expected):
    # Preserve the invoked POSIX venv path: its symlink resolves to base Python.
    if Path(executable).absolute() != Path(expected).absolute():
        raise ValueError("download service must use the installed interpreter")
"""
exec(LAUNCHER_VALIDATOR)

INSTALLED_PROBE = (
    ORIGIN_VALIDATOR
    + """
import importlib
import importlib.metadata as metadata
import json
import platform

distribution = metadata.distribution("ai-model-explorer")
recorded = {Path(distribution.locate_file(item)).resolve() for item in distribution.files or []}
modules = (
    "main", "cli", "api_server", "config", "tui_app", "app.viewer", "app.startup_viewer",
    "app.modals", "core.scoring", "core.diagnostics", "core.hardware", "core.exports",
    "downloads.service_client", "downloads.download_service", "providers.hf_provider",
    "providers.ollama_provider", "providers.lmstudio_provider", "providers.docker_provider",
    "providers.mlx_provider", "results.results_presenter", "search.search_orchestration", "terminal_ui",
)
origins = {}
for name in modules:
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    assert_installed_origin(origin, Path(sys.prefix), Path(sys.argv[1]), recorded)
    origins[name] = origin.relative_to(Path(sys.prefix)).as_posix()
print(json.dumps({
    "python": platform.python_version(), "platform": platform.platform(),
    "app_version": distribution.version, "module_origins": origins,
    "dependencies": sorted([{"name": item.metadata["Name"], "version": item.version}
                            for item in metadata.distributions()], key=lambda item: item["name"].lower()),
}))
"""
)

SERVICE_CLIENT_PROBE = (
    LAUNCHER_VALIDATOR
    + """
import subprocess
import sys
from pathlib import Path
from downloads import service_client

owned = []
original = subprocess.Popen

def start_owned(args, **kwargs):
    expected = Path(sys.executable)
    if sys.platform.startswith("win"):
        expected = expected.with_name("pythonw.exe")
    assert_launcher_interpreter(args[0], expected)
    process = original(args, **kwargs)
    owned.append(process)
    return process

service_client.subprocess.Popen = start_owned
try:
    service_client._start_service_process()
    if len(owned) != 1 or owned[0].wait(timeout=20) != 0:
        raise RuntimeError("installed download-service launch failed")
finally:
    for process in owned:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
print("installed-service-client-launch-ok")
"""
)


def select_artifact(dist_dir: Path, kind: str) -> Path:
    candidates = sorted(dist_dir.glob("*.whl" if kind == "wheel" else "*.tar.gz"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {kind} artifact, found {len(candidates)}")
    return candidates[0]


def isolated_environment(runtime: Path) -> dict[str, str]:
    """Preserve OS/pip transport settings while isolating application configuration."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("AIMODEL_", "HF_", "HUGGINGFACE_", "PYTHON"))
        and key.upper()
        not in {
            "LMSTUDIO_HOST",
            "DOCKER_MODEL_RUNNER_HOST",
            "PIP_TARGET",
            "PIP_PREFIX",
            "PIP_USER",
            "VIRTUAL_ENV",
        }
    }
    env.update(
        {
            "AIMODEL_SMOKE": "1",
            "AIMODEL_CACHE_DB_PATH": str(runtime / "cache.db"),
            "AIMODEL_DOWNLOAD_DB_PATH": str(runtime / "downloads.db"),
            "AIMODEL_HF_MODELS_DIR": str(runtime / "models"),
            "AIMODEL_DOWNLOAD_SERVICE_HOST": "127.0.0.1",
            "AIMODEL_DOWNLOAD_SERVICE_TOKEN": "package-smoke-owned-token",
            "AIMODEL_OLLAMA_API_BASE": "http://127.0.0.1:1",
            "LMSTUDIO_HOST": "http://127.0.0.1:1",
            "DOCKER_MODEL_RUNNER_HOST": "http://127.0.0.1:1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CONFIG_FILE": os.devnull,
        }
    )
    return env


def run_step(name: str, command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> str:
    print(f"[package] {name}", flush=True)
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        # Bounded diagnostic tail; never mistake a failed stage for a successful smoke.
        print((result.stdout + result.stderr)[-6000:], file=sys.stderr)
        result.check_returncode()
    return result.stdout


def verify_cli_exports(executable: Path, cwd: Path, env: dict[str, str]) -> None:
    plan = json.loads(run_step("installed hardware-plan JSON",
                               [str(executable), "plan", "sample-7b", "--json"], cwd, env, 15))
    if (plan.get("schema_version") != 1 or plan.get("kind") != "hardware_plan"
        or plan.get("artifact_metadata") is not None or not plan.get("plans")):
        raise ValueError("installed hardware-plan export contract failed")
    saved = cwd / "saved-search.json"
    saved.write_text(json.dumps({"schema_version": 1, "kind": "search", "models": [
        {"id": "one", "name": "one", "source": "Hugging Face", "scores": {"composite": 1}},
        {"id": "two", "name": "two", "source": "Ollama", "scores": {"composite": 2}}],
        "errors": []}), encoding="utf-8")
    comparison = json.loads(run_step("installed saved comparison JSON",
                                     [str(executable), "compare", "--input", str(saved),
                                      "two", "one", "--json"], cwd, env, 15))
    if (comparison.get("schema_version") != 1 or comparison.get("kind") != "comparison"
        or [model.get("id") for model in comparison.get("models", [])] != ["two", "one"]
        or comparison["models"][0].get("scores", {}).get("composite") != 2):
        raise ValueError("installed comparison export contract failed")


def verify_package(artifact: Path, checkout: Path) -> dict:
    artifact, checkout = artifact.resolve(), checkout.resolve()
    with tempfile.TemporaryDirectory(prefix="ai-model-installed-") as temporary:
        root = Path(temporary).resolve()
        if root.is_relative_to(checkout):
            raise ValueError("verification working directory must be outside the checkout")
        prefix, cwd, runtime = root / "environment", root / "work", root / "runtime"
        cwd.mkdir()
        runtime.mkdir()
        env = isolated_environment(runtime)
        # Fresh environments never inherit checkout/dev dependencies or editable installs.
        venv.EnvBuilder(with_pip=True).create(prefix)
        scripts = prefix / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        run_step(
            "install artifact with declared dependencies",
            [str(python), "-I", "-m", "pip", "install", str(artifact)],
            cwd,
            env,
            600,
        )
        run_step("dependency consistency", [str(python), "-I", "-m", "pip", "check"], cwd, env, 30)
        installed = json.loads(
            run_step(
                "installed module origins",
                [str(python), "-I", "-c", INSTALLED_PROBE, str(checkout)],
                cwd,
                env,
                30,
            )
        )
        suffix = ".exe" if os.name == "nt" else ""
        output = run_step(
            "installed CLI entry point",
            [str(scripts / ("ai-model-explorer-cli" + suffix)), "version"],
            cwd,
            env,
            15,
        )
        if f"AI Model Explorer v{installed['app_version']}" not in output:
            raise ValueError("installed CLI version does not match artifact metadata")
        doctor = json.loads(
            run_step(
                "installed offline doctor",
                [
                    str(scripts / ("ai-model-explorer-cli" + suffix)),
                    "doctor",
                    "--offline",
                    "--json",
                ],
                cwd,
                env,
                15,
            )
        )
        if doctor.get("offline") is not True or doctor.get("status") == "error":
            raise ValueError("installed offline doctor failed")
        verify_cli_exports(scripts / ("ai-model-explorer-cli" + suffix), cwd, env)
        run_step(
            "installed TUI entry point",
            [str(scripts / ("ai-model-explorer" + suffix))],
            cwd,
            env,
            45,
        )
        run_step("installed REST health", [str(python), "-I", "-m", "api_server"], cwd, env, 20)
        run_step(
            "installed download-service health and authenticated jobs",
            [str(python), "-I", "-m", "downloads.download_service"],
            cwd,
            env,
            20,
        )
        run_step(
            "installed client subprocess launch",
            [str(python), "-I", "-c", SERVICE_CLIENT_PROBE],
            cwd,
            env,
            30,
        )
        return {
            "schema_version": 1,
            "status": "passed",
            "artifact": artifact.name,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "outside_checkout": True,
            "fresh_environment": True,
            "pythonpath_removed": True,
            **installed,
            "checks": [
                "pip_check",
                "module_origins",
                "cli_entry",
                "offline_doctor",
                "hardware_plan_json",
                "saved_comparison_json",
                "tui_entry",
                "rest_health",
                "download_service_health_jobs",
                "client_subprocess_launch",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("wheel", "sdist"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"schema_version": 1, "status": "running", "artifact_kind": args.kind}) + "\n",
        encoding="utf-8",
    )
    try:
        artifact = select_artifact(args.dist_dir.resolve(), args.kind)
        report = verify_package(artifact, Path(__file__).resolve().parents[1])
        report["artifact_kind"] = args.kind
    except Exception as exc:
        args.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "artifact_kind": args.kind,
                    "error_type": type(exc).__name__,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[package] {args.kind} passed: {report['app_version']} / Python {report['python']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
