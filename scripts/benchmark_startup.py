"""Opt-in fresh-process startup measurements; no model acquisition or search."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

LIMITS_MS = {"startup_ms": 2000, "input_ms": 100, "providers_ms": 2000}


def emit_line(line):
    """Send the measurement protocol to the controlling process."""
    sys.__stdout__.write(line + "\n")
    sys.__stdout__.flush()


def summarize_samples(samples, expected_runs):
    """Use nearest-rank p95; incomplete or invalid evidence cannot pass."""
    valid = [
        sample
        for sample in samples
        if sample.get("status") == "ok"
        and all(
            isinstance(sample.get(key), (int, float))
            and not isinstance(sample[key], bool)
            and math.isfinite(sample[key])
            and sample[key] >= 0
            for key in LIMITS_MS
        )
    ]
    metrics = {}
    for key, limit in LIMITS_MS.items():
        values = sorted(sample[key] for sample in valid)
        metrics[key] = {
            "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
            "max": values[-1] if values else None,
            "limit": limit,
        }
    return {
        "successful_runs": len(valid),
        "failed_runs": len(samples) - len(valid),
        "metrics": metrics,
        "passed": expected_runs >= 20
        and len(valid) == len(samples) == expected_runs
        and all(metric["p95"] <= metric["limit"] for metric in metrics.values()),
    }


def collect_sample(command, timeout, *, env=None, cwd=None):
    """Read one complete sample and stop only the child process owned by this run."""
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    messages = queue.Queue()

    def read_stdout():
        for line in process.stdout:
            if line.startswith(("STARTUP_READY", "STARTUP_SAMPLE ")):
                messages.put((time.perf_counter(), line))
        messages.put((time.perf_counter(), None))

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    startup_ms = None
    try:
        deadline = started + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return {"status": "error", "code": "child_timeout"}
            try:
                received_at, line = messages.get(timeout=remaining)
            except queue.Empty:
                return {"status": "error", "code": "child_timeout"}
            if line is None:
                return {"status": "error", "code": "child_exited_without_sample"}
            if line.startswith("STARTUP_READY"):
                startup_ms = (received_at - started) * 1000
                continue
            try:
                result = json.loads(line.removeprefix("STARTUP_SAMPLE "))
            except json.JSONDecodeError:
                return {"status": "error", "code": "invalid_sample"}
            if not isinstance(result, dict):
                return {"status": "error", "code": "invalid_sample"}
            if startup_ms is not None:
                result["startup_ms"] = round(startup_ms, 3)
            return result
    finally:
        # Optional probes may still be waiting on upstream timeouts. Their completion
        # is outside these response-path metrics; no shared runtime/service is stopped.
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=1)
        process.stdout.close()


async def measure_child():
    """Exercise real runtime startup, input dispatch and a loopback REST request."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from textual.widgets import Input

    import api_server
    from app.viewer import AIModelViewer

    app = AIModelViewer()
    input_started = None
    input_ms = None
    provider_pending = None

    def on_message(message):
        nonlocal input_ms, provider_pending
        if (
            isinstance(message, Input.Changed)
            and message.input.id == "search-input"
            and message.value == "x"
            and input_started is not None
            and input_ms is None
        ):
            input_ms = (time.perf_counter() - input_started) * 1000
            provider_pending = app._provider_filter_refresh_running

    async with app.run_test(size=(120, 40), message_hook=on_message) as pilot:
        emit_line("STARTUP_READY")
        app.query_one("#search-input", Input).focus()
        await pilot.pause()
        input_started = time.perf_counter()
        await pilot.press("x")
        if input_ms is None:
            raise RuntimeError("input change was not observed")
        deadline = time.monotonic() + 15
        while app.latest_specs is None and time.monotonic() < deadline:
            await asyncio.sleep(0.025)
        if app.latest_specs is None:
            raise RuntimeError("hardware snapshot was not delivered")
        server, _thread = api_server.start_server_background(port=0, monitor=app.monitor)
        try:
            import urllib.request

            started = time.perf_counter()
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/api/v1/providers", timeout=2
            ) as response:
                payload = json.load(response)
            providers_ms = (time.perf_counter() - started) * 1000
            result = {
                "status": "ok",
                "input_ms": round(input_ms, 3),
                "providers_ms": round(providers_ms, 3),
                "input_during_provider_probe": provider_pending,
                "rest_discovery": payload["discovery"],
                "cpu": app.latest_specs["cpu_name"],
                "gpu": app.latest_specs["gpu_name"],
            }
        finally:
            server.shutdown()
            server.server_close()
        emit_line("STARTUP_SAMPLE " + json.dumps(result, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--machine", default="local-machine")
    parser.add_argument("--output", type=Path, default=Path(".venv/startup-benchmark.json"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        try:
            asyncio.run(measure_child())
        except Exception as exc:
            print(
                "STARTUP_SAMPLE "
                + json.dumps(
                    {
                        "status": "error",
                        "code": "measurement_failed",
                        "error_type": type(exc).__name__,
                    }
                ),
                flush=True,
            )
            return 1
        return 0
    if args.runs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("runs and timeout must be positive")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    work_dir = root / ".venv" / "startup-benchmark-runs" / str(time.time_ns())
    samples = []
    for index in range(args.runs):
        run_dir = work_dir / str(index + 1)
        run_dir.mkdir(parents=True)
        env = os.environ.copy()
        env.pop("AIMODEL_SMOKE", None)
        env.update(
            {
                "AIMODEL_CACHE_DB_PATH": str(run_dir / "cache.db"),
                "AIMODEL_DOWNLOAD_DB_PATH": str(run_dir / "downloads.db"),
                "AIMODEL_HF_MODELS_DIR": str(run_dir / "models"),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        sample = collect_sample(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            timeout=args.timeout,
            env=env,
            cwd=root,
        )
        samples.append(sample)
        print(f"Run {index + 1}/{args.runs}: {sample['status']}", flush=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report = {
        "schema_version": 1,
        "machine": args.machine,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_revision": revision,
        "scenario": "fresh-process-cold-cache",
        "terminal_size": [120, 40],
        "samples": samples,
        "summary": summarize_samples(samples, args.runs),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)
    return 0 if report["summary"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
