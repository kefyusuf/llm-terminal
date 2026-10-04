"""Measurement failures and tail latency must not become false passing evidence."""

from __future__ import annotations

import json
import sys

from scripts.benchmark_startup import collect_sample, summarize_samples


def sample(startup_ms=300, input_ms=20, providers_ms=50):
    return {
        "status": "ok",
        "startup_ms": startup_ms,
        "input_ms": input_ms,
        "providers_ms": providers_ms,
    }


def test_summary_uses_nearest_rank_p95_and_retains_slow_maximum():
    samples = [sample(startup_ms=100 * i) for i in range(1, 20)] + [sample(startup_ms=9000)]
    report = summarize_samples(samples, expected_runs=20)
    assert report["metrics"]["startup_ms"]["p95"] == 1900
    assert report["metrics"]["startup_ms"]["max"] == 9000
    assert report["passed"] is True


def test_failed_or_insufficient_runs_cannot_pass_a_release_budget():
    samples = [sample() for _ in range(19)] + [{"status": "error", "code": "child_timeout"}]
    report = summarize_samples(samples, expected_runs=20)
    assert report["passed"] is False
    assert report["successful_runs"] == 19
    assert report["failed_runs"] == 1
    assert summarize_samples([sample()], expected_runs=1)["passed"] is False


def test_slow_input_and_invalid_metrics_fail_the_budget():
    assert summarize_samples([sample(input_ms=101) for _ in range(20)], 20)["passed"] is False
    report = summarize_samples(
        [sample() for _ in range(19)] + [sample(startup_ms=float("nan"))], 20
    )
    assert report["successful_runs"] == 19
    assert report["failed_runs"] == 1
    assert report["passed"] is False


def test_collection_returns_completed_measurement_without_waiting_for_probe_cleanup():
    payload = json.dumps(sample())
    script = f"import time; print('STARTUP_SAMPLE ' + {payload!r}, flush=True); time.sleep(30)"
    result = collect_sample([sys.executable, "-c", script], timeout=2)
    assert result["status"] == "ok"
    assert result["startup_ms"] == 300
    assert result["input_ms"] == 20
    assert result["providers_ms"] == 50


def test_collection_marks_missing_measurement_as_failure():
    result = collect_sample([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.1)
    assert result == {"status": "error", "code": "child_timeout"}


def test_protocol_reaches_parent_while_textual_captures_stdout():
    payload = json.dumps(sample())
    script = f"""
import asyncio
from textual.app import App
from scripts.benchmark_startup import emit_line
async def run():
    async with App().run_test():
        emit_line('STARTUP_SAMPLE ' + {payload!r})
        await asyncio.sleep(30)
asyncio.run(run())
"""
    result = collect_sample([sys.executable, "-c", script], timeout=4)
    assert result["status"] == "ok"
    assert result["input_ms"] == 20
