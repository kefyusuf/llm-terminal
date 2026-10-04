"""A forcibly stopped parent must not leave the owned HF child writing."""
import subprocess
import sys
import time
from pathlib import Path

import psutil


def active(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_watchdog_does_not_break_normal_python_shutdown():
    process = subprocess.Popen(
        [sys.executable, "-c", "from downloads.parent_lifetime import start_parent_watchdog; "
         "import time; start_parent_watchdog(); time.sleep(0.1)"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        process.wait(timeout=5)
        assert process.returncode == 0, process.stderr.read().decode(errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        process.stdin.close()
        process.stderr.close()


def test_hf_script_exits_after_parent_is_forcibly_stopped(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "huggingface_hub.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "def hf_hub_download(**kwargs):\n"
        "    Path(kwargs['local_dir'], 'ready').write_text(str(os.getpid()))\n"
        "    while True: time.sleep(0.05)\n", encoding="utf-8"
    )
    root = Path(__file__).resolve().parents[1]
    code = (
        "import os, subprocess, sys, time\n"
        "from downloads.runner import _hf_download_script\n"
        "env = os.environ.copy()\n"
        "env['PYTHONPATH'] = sys.argv[1] + os.pathsep + sys.argv[2]\n"
        "child = subprocess.Popen([sys.executable, '-c', _hf_download_script(),"
        "'owner/repo', 'model.gguf', sys.argv[3], '', '', '', 'parent-stdin'],"
        "stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)\n"
        "while True: time.sleep(0.05)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", code, str(fixture), str(root), str(tmp_path)],
                              cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    try:
        deadline = time.monotonic() + 10
        ready = tmp_path / "ready"
        child_pid = None
        while time.monotonic() < deadline and parent.poll() is None:
            if ready.exists():
                value = ready.read_text()
                if value.isdigit():
                    child_pid = int(value)
                    break
            time.sleep(0.02)
        assert child_pid is not None, "the exact HF child did not reach the fixture download"
        child = psutil.Process(child_pid)
        parent.kill()
        parent.wait(timeout=3)
        deadline = time.monotonic() + 3
        while active(child) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not active(child), "owned HF child survived parent death"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        if child is not None and active(child):
            # psutil checks this recorded process identity before termination.
            child.kill()
            child.wait(timeout=3)
