from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX = (3, 14)
LOCK_PYTHON = (3, 12)
REQUIREMENTS_DIRNAME = "requirements"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def requirements_dir(root: Path) -> Path:
    candidate = root / REQUIREMENTS_DIRNAME
    if candidate.exists():
        return candidate
    return root


def requirement_path(root: Path, name: str) -> Path:
    return requirements_dir(root) / name


def ensure_supported_python(version_info: tuple[int, ...] | None = None) -> None:
    version = version_info or tuple(sys.version_info)
    major_minor = version[:2]
    if major_minor < SUPPORTED_PYTHON_MIN or major_minor > SUPPORTED_PYTHON_MAX:
        raise SystemExit("scripts/dev.py requires Python 3.10-3.14")


def select_lock_targets(system_name: str | None = None) -> list[tuple[str, str]]:
    resolved_system = system_name or platform.system()
    if resolved_system == "Windows":
        return [
            ("requirements.in", "requirements-windows.txt"),
            ("requirements-dev.in", "requirements-dev-windows.txt"),
        ]
    if resolved_system == "Linux":
        return [
            ("requirements.in", "requirements-linux.txt"),
            ("requirements-dev.in", "requirements-dev-linux.txt"),
        ]
    raise SystemExit(f"Unsupported platform for lock management: {resolved_system}")


def select_dev_lock(system_name: str | None = None) -> str:
    return select_lock_targets(system_name)[1][1]


def venv_python_path(root: Path, system_name: str | None = None) -> Path:
    resolved_system = system_name or platform.system()
    if resolved_system == "Windows":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def read_venv_python_version(venv_dir: Path) -> tuple[int, int] | None:
    config_path = venv_dir / "pyvenv.cfg"
    if not config_path.exists():
        return None

    for line in config_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("version ="):
            continue

        raw_version = line.split("=", 1)[1].strip()
        parts = raw_version.split(".")
        if len(parts) < 2:
            return None

        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    return None


def ensure_virtualenv(root: Path, system_name: str | None = None) -> Path:
    venv_dir = root / ".venv"
    expected_python = venv_python_path(root, system_name)
    expected_version = tuple(sys.version_info[:2])
    existing_version = read_venv_python_version(venv_dir)
    needs_recreate = venv_dir.exists() and (
        not expected_python.exists() or (existing_version is not None and existing_version != expected_version)
    )

    if needs_recreate:
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, cwd=root)
    return venv_dir


def ensure_pip(venv_python: Path, root: Path) -> None:
    result = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    subprocess.run([str(venv_python), "-m", "ensurepip", "--upgrade"], check=True, cwd=root)


def install_lockfile(venv_python: Path, lockfile: Path, root: Path) -> None:
    if not lockfile.exists():
        raise SystemExit(f"Missing committed lock file: {lockfile.name}")

    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(lockfile)],
        check=True,
        cwd=root,
    )


def bootstrap() -> int:
    ensure_supported_python()
    root = project_root()
    system_name = platform.system()
    ensure_virtualenv(root, system_name)
    venv_python = venv_python_path(root, system_name)
    ensure_pip(venv_python, root)
    lockfile = requirement_path(root, select_dev_lock(system_name))
    install_lockfile(venv_python, lockfile, root)
    print(f"[bootstrap] installed from {lockfile.relative_to(root).as_posix()}")
    return 0


def compile_lock(source_file: Path, output_file: Path, cwd: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            f"--output-file={output_file.relative_to(cwd).as_posix()}",
            source_file.relative_to(cwd).as_posix(),
        ],
        check=True,
        cwd=cwd,
    )


def normalize_lock_text(text: str) -> str:
    lines = text.splitlines()
    start_index = 0

    for index, line in enumerate(lines):
        if not line or line.startswith("#"):
            continue
        start_index = index
        break
    else:
        return ""

    return "\n".join(lines[start_index:]).strip()


def check_lock_targets(root: Path, system_name: str | None = None) -> None:
    for source_name, output_name in select_lock_targets(system_name):
        output_file = requirement_path(root, output_name)

        if not output_file.exists():
            raise SystemExit(f"Missing committed lock file: {output_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            temp_requirements = temp_root / REQUIREMENTS_DIRNAME
            temp_requirements.mkdir(exist_ok=True)
            for input_name, _output_name in select_lock_targets(system_name):
                shutil.copy2(requirement_path(root, input_name), temp_requirements / input_name)
            for _input_name, existing_output_name in select_lock_targets(system_name):
                committed_output = requirement_path(root, existing_output_name)
                if committed_output.exists():
                    shutil.copy2(committed_output, temp_requirements / existing_output_name)

            generated_file = temp_requirements / output_name
            compile_lock(temp_requirements / source_name, generated_file, temp_root)
            generated_text = normalize_lock_text(generated_file.read_text(encoding="utf-8"))

        committed_text = normalize_lock_text(output_file.read_text(encoding="utf-8"))
        if committed_text != generated_text:
            raise SystemExit(f"Lock file is stale: {output_name}")


def lock(*, check: bool = False) -> int:
    ensure_supported_python()
    root = project_root()
    system_name = platform.system()
    current_version = tuple(sys.version_info[:2])

    if current_version != LOCK_PYTHON:
        current_label = f"{current_version[0]}.{current_version[1]}"
        lock_label = f"{LOCK_PYTHON[0]}.{LOCK_PYTHON[1]}"
        if check:
            print(
                f"[lock] skipping freshness check on Python {current_label}; canonical lock runtime is Python {lock_label}"
            )
            return 0
        raise SystemExit(f"[lock] regenerate committed lock files with Python {lock_label}")

    if check:
        check_lock_targets(root, system_name)
        print(f"[lock] {system_name} lock files are fresh")
        return 0

    for source_name, output_name in select_lock_targets(system_name):
        compile_lock(requirement_path(root, source_name), requirement_path(root, output_name), root)
        print(f"[lock] regenerated {REQUIREMENTS_DIRNAME}/{output_name}")

    return 0


def run_checked_step(step_name: str, command: list[str], root: Path) -> None:
    try:
        subprocess.run(command, check=True, cwd=root)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[verify] {step_name} failed") from exc


def verify() -> int:
    ensure_supported_python()
    root = project_root()
    venv_python = venv_python_path(root)

    if not venv_python.exists():
        raise SystemExit("[verify] missing virtualenv; run python scripts/dev.py bootstrap first")

    run_checked_step("pytest", [str(venv_python), "-m", "pytest", "-q"], root)
    run_checked_step(
        "import smoke",
        [str(venv_python), "-c", "import main, app; print('import-ok')"],
        root,
    )
    run_checked_step("ruff", [str(venv_python), "-m", "ruff", "check", "."], root)
    return 0


def run_smoke_step(
    step_name: str,
    command: list[str],
    root: Path,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
) -> None:
    try:
        subprocess.run(command, check=True, cwd=root, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"[smoke] {step_name} timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[smoke] {step_name} failed") from exc


def smoke() -> int:
    ensure_supported_python()
    root = project_root()
    venv_python = venv_python_path(root)

    if not venv_python.exists():
        raise SystemExit("[smoke] missing virtualenv; run python scripts/dev.py bootstrap first")

    smoke_env = os.environ.copy()
    smoke_env["AIMODEL_SMOKE"] = "1"
    tui_timeout = 45

    run_smoke_step("cli", [str(venv_python), "-m", "cli", "system"], root, timeout=10)
    run_smoke_step(
        "api",
        [str(venv_python), "-m", "api_server"],
        root,
        timeout=15,
        env=smoke_env,
    )
    run_smoke_step(
        "tui",
        [str(venv_python), "main.py"],
        root,
        timeout=tui_timeout,
        env=smoke_env,
    )
    run_smoke_step(
        "download-service",
        [str(venv_python), "-m", "downloads.download_service"],
        root,
        timeout=15,
        env=smoke_env,
    )
    return 0


def not_implemented(command_name: str) -> int:
    raise SystemExit(f"scripts/dev.py {command_name} is not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project development helper commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap", help="Create or reuse .venv and install the platform dev lock")
    subparsers.add_parser("verify", help="Run the project verify lane")
    subparsers.add_parser("smoke", help="Run the project smoke lane")
    lock_parser = subparsers.add_parser("lock", help="Manage committed lock files")
    lock_parser.add_argument("--check", action="store_true", help="Fail if the current-platform locks are missing or stale")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        return bootstrap()
    if args.command == "verify":
        return verify()
    if args.command == "smoke":
        return smoke()
    if args.command == "lock":
        return lock(check=args.check)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
