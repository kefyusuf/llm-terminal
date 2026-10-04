"""Calibration protocol tests are simulated, never real benchmark evidence."""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("url", ["https://ollama.com", "http://user:secret@127.0.0.1:11434",
                                "http://127.0.0.1:11434/path", "http://127.0.0.1:11434?key=value"])
def test_calibration_refuses_nonlocal_or_credentialed_endpoint(url):
    from scripts.calibrate_ollama import validate_endpoint

    with pytest.raises(ValueError, match="loopback"):
        validate_endpoint(url)


def response():
    return {"model": "fixture:latest", "done": True, "prompt_eval_count": 20,
            "prompt_eval_cached_count": 5, "prompt_eval_duration": 1000000000,
            "eval_count": 4, "eval_duration": 2000000000,
            "load_duration": 100000000, "total_duration": 4000000000}


def test_prompt_generation_and_wall_metrics_are_distinct():
    from scripts.calibrate_ollama import sample_metrics

    result = sample_metrics(response(), wall_seconds=4.2, max_tokens=8)
    assert result["prompt_tok_s"] == 15
    assert result["generation_tok_s"] == 2
    assert result["wall_seconds"] == 4.2
    assert result["runtime_total_seconds"] == 4
    assert result["load_seconds"] == .1


@pytest.mark.parametrize("change", [{"done": False}, {"eval_duration": 0}, {"eval_count": True},
                                    {"eval_count": 9}, {"remote_host": "https://ollama.com"}])
def test_invalid_or_remote_generation_cannot_become_a_sample(change):
    from scripts.calibrate_ollama import sample_metrics

    with pytest.raises(ValueError):
        sample_metrics({**response(), **change}, wall_seconds=4.2, max_tokens=8)


def test_summary_reports_dispersion_and_prediction_error():
    from scripts.calibrate_ollama import summarize_samples

    result = summarize_samples([{"generation_tok_s": 2, "prompt_tok_s": 10, "wall_seconds": 1},
                                {"generation_tok_s": 4, "prompt_tok_s": 20, "wall_seconds": 3}], prediction=6)
    assert result["generation_tok_s"]["median"] == 3
    assert result["generation_tok_s"]["min"] == 2
    assert result["generation_tok_s"]["max"] == 4
    assert result["generation_tok_s"]["population_stddev"] == 1
    assert result["prediction_error_on_median_percent"] == 100


@pytest.mark.parametrize("status", [{}, {"cloud": {"disabled": False}}, {"cloud": {"disabled": "true"}}])
def test_localhost_alone_is_not_proof_that_cloud_features_are_disabled(status):
    from scripts.calibrate_ollama import require_local_runtime

    with pytest.raises(ValueError, match="cloud"):
        require_local_runtime(status)


@pytest.mark.parametrize("change_digest", [False, True])
def test_protocol_uses_exact_installed_model_and_preserves_identity(monkeypatch, change_digest):
    from scripts import calibrate_ollama as calibration

    calls = []
    entry = {"name": "fixture:latest", "digest": "a" * 64, "size": 1024**3}

    def request(base, path, body=None, **kwargs):
        calls.append((path, body))
        if path == "/api/version":
            return {"version": "fixture-runtime"}
        if path == "/api/status":
            return {"cloud": {"disabled": True}}
        if path == "/api/tags":
            count = sum(path == "/api/tags" for path, _ in calls)
            return {"models": [{**entry, "digest": "b" * 64 if change_digest and count > 1 else "a" * 64}]}
        if path == "/api/show":
            return {"capabilities": ["completion"], "model_info": {}}
        if path == "/api/generate":
            return response()
        raise AssertionError("unexpected API operation")

    monkeypatch.setattr(calibration, "request_json", request)
    ticks = iter(range(1000))
    monkeypatch.setattr(calibration.time, "monotonic", lambda: next(ticks) * .01)
    monkeypatch.setattr(calibration, "hardware_specs", lambda: {"cpu_name": "fixture", "cpu_cores": 1,
                       "ram_total": 8, "ram_free": 8, "gpu_name": "fixture", "has_gpu": False,
                       "vram_free": 0, "vram_total": 0, "gpu_vendor": "none", "backend": "cpu"})
    args = SimpleNamespace(url="http://127.0.0.1:11434", model="fixture:latest", repetitions=2,
                           tokens=8, context=512, max_seconds=30, request_timeout=5)
    if change_digest:
        with pytest.raises(ValueError, match="identity changed"):
            calibration.run_trial(args)
        return
    report = calibration.run_trial(args)
    assert report["kind"] == "runtime_calibration"
    assert report["model_facts"]["model_digest"] == "a" * 64
    assert len(report["samples"]) == 2
    assert len([path for path, _ in calls if path == "/api/generate"]) == 3  # warmup + samples
    assert all(body["stream"] is False and body["options"]["num_predict"] == 8
               for path, body in calls if path == "/api/generate")
    assert not any(path in {"/api/pull", "/api/create", "/api/delete"} for path, _ in calls)


@pytest.mark.parametrize("entries", [[], [{"name": "fixture:latest", "digest": "a" * 64, "size": 100,
                                          "remote_model": "upstream"}],
    [{"name": "fixture:latest", "digest": "a" * 64, "size": 100}] * 2])
def test_missing_remote_and_ambiguous_model_entries_are_rejected(entries):
    from scripts.calibrate_ollama import model_entry

    with pytest.raises(ValueError):
        model_entry({"models": entries}, "fixture:latest")


def test_non_finite_prediction_is_rejected():
    from scripts.calibrate_ollama import summarize_samples

    with pytest.raises(ValueError, match="finite"):
        summarize_samples([{"generation_tok_s": 2}] * 2, prediction=float("nan"))


def test_cli_protocol_with_real_local_http_fixture(tmp_path):
    operations = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data):
            encoded = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            operations.append(self.path)
            data = {"/api/version": {"version": "simulated-fixture"},
                    "/api/status": {"cloud": {"disabled": True}},
                    "/api/tags": {"models": [{"name": "fixture:latest", "digest": "a" * 64, "size": 1024**3}]}}
            self.reply(data[self.path])

        def do_POST(self):
            operations.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "fixture:latest"
            if self.path == "/api/show":
                self.reply({"capabilities": ["completion"], "model_info": {}})
            elif self.path == "/api/generate":
                assert body["stream"] is False
                assert body["options"]["num_predict"] == 8
                self.reply(response())
            else:
                raise AssertionError("unexpected calibration operation")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report_path = tmp_path / "simulated-report.json"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(AIMODEL_CACHE_DB_PATH=str(tmp_path / "cache.db"),
               AIMODEL_DOWNLOAD_DB_PATH=str(tmp_path / "jobs.db"), AIMODEL_HF_MODELS_DIR=str(tmp_path / "models"))
    try:
        result = subprocess.run([sys.executable, str(root / "scripts/calibrate_ollama.py"), "fixture:latest",
                  "--url", f"http://127.0.0.1:{server.server_port}", "--repetitions", "2", "--tokens", "8",
                  "--max-seconds", "30", "--output", str(report_path)], cwd=root, env=env,
                  capture_output=True, text=True, timeout=40)
        assert result.returncode == 0, result.stderr
        report = json.loads(report_path.read_text())
        assert report["runtime"]["version"] == "simulated-fixture"
        assert len(report["samples"]) == 2
        assert operations.count("/api/generate") == 3
        assert "response" not in report
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cli_failure_explains_boundary_without_writing_result(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "unused.json"
    result = subprocess.run([sys.executable, str(root / "scripts/calibrate_ollama.py"), "fixture:latest",
              "--url", "https://ollama.com", "--output", str(output), "--max-seconds", "5"],
              cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "loopback" in result.stderr
    assert not output.exists()


def test_supervisor_enforces_wall_budget_against_a_slow_http_body(tmp_path):
    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            data = json.dumps({"version": "slow-fixture-version"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                for byte in data:
                    if stop.wait(.6):
                        break
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "unused.json"
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    try:
        result = subprocess.run([sys.executable, str(root / "scripts/calibrate_ollama.py"), "fixture:latest",
                  "--url", f"http://127.0.0.1:{server.server_port}", "--max-seconds", "5",
                  "--request-timeout", "30", "--output", str(output)], cwd=root,
                  capture_output=True, text=True, timeout=15)
        assert result.returncode != 0
        assert "deadline" in result.stderr
        assert time.monotonic() - started < 12
        assert not output.exists()
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
