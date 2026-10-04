"""Opt-in bounded calibration against an installed model on a dedicated local Ollama."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT = "Explain in three short sentences how a hash helps verify a downloaded file."


class CalibrationError(ValueError):
    """A fixed public diagnostic, never raw runtime or transport error text."""


def validate_endpoint(url):
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
        or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
        raise CalibrationError("calibration requires a credential-free loopback HTTP endpoint")
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{parsed.port or 11434}"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(base, path, body=None, *, deadline, request_timeout):
    if path not in {"/api/version", "/api/status", "/api/tags", "/api/show", "/api/generate"}:
        raise CalibrationError("unsupported calibration operation")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("calibration deadline exceeded")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    req = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                  headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=min(request_timeout, remaining)) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise CalibrationError("runtime JSON exceeds the 1 MiB limit")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("error"):
        raise CalibrationError("runtime did not return a valid calibration response")
    return data


def require_local_runtime(status):
    cloud = status.get("cloud")
    if not isinstance(cloud, dict) or cloud.get("disabled") is not True:
        raise CalibrationError("runtime cloud features must be verifiably disabled before calibration")


def reject_remote(data):
    if data.get("remote_model") or data.get("remote_host"):
        raise CalibrationError("remote models are excluded from local calibration")


def model_entry(tags, model):
    entries = tags.get("models")
    if not isinstance(entries, list) or len(entries) > 1000:
        raise CalibrationError("runtime model listing is unsupported")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == model]
    if len(matches) != 1 or model.endswith((":cloud", "-cloud")):
        raise CalibrationError("select one exact installed local model name")
    entry = matches[0]
    reject_remote(entry)
    if (not isinstance(entry.get("digest"), str) or not re.fullmatch(r"[a-fA-F0-9]{64}", entry["digest"])
        or type(entry.get("size")) is not int or entry["size"] <= 0):
        raise CalibrationError("installed model identity or disk size is unavailable")
    return entry


def sample_metrics(response, *, wall_seconds, max_tokens):
    reject_remote(response)
    if response.get("done") is not True or response.get("error"):
        raise CalibrationError("incomplete generation cannot become a calibration sample")

    def integer(name, *, positive=False):
        value = response.get(name)
        if type(value) is not int or value < (1 if positive else 0):
            raise CalibrationError("runtime token/timing metric is unsupported")
        return value

    count, duration = integer("eval_count", positive=True), integer("eval_duration", positive=True)
    if count > max_tokens or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise CalibrationError("generation token/time bound was not respected")
    prompt_count = integer("prompt_eval_count")
    prompt_duration = integer("prompt_eval_duration")
    cached = integer("prompt_eval_cached_count") if "prompt_eval_cached_count" in response else None
    if cached is not None and cached > prompt_count:
        raise CalibrationError("runtime cached-token metric is inconsistent")
    evaluated = prompt_count - (cached or 0)
    return {"generation_tok_s": count * 1e9 / duration,
            "prompt_tok_s": evaluated * 1e9 / prompt_duration if prompt_duration else None,
            "prompt_cache_count_reported": cached is not None,
            "prompt_eval_count": prompt_count, "prompt_cached_count": cached,
            "generation_count": count, "wall_seconds": wall_seconds,
            "runtime_total_seconds": integer("total_duration", positive=True) / 1e9,
            "load_seconds": integer("load_duration") / 1e9 if "load_duration" in response else None}


def summarize_samples(samples, *, prediction):
    if len(samples) < 2 or type(prediction) not in (int, float) or not math.isfinite(prediction) or prediction < 0:
        raise CalibrationError("calibration summary requires repeated samples and a finite prediction")
    result = {}
    for metric in ("generation_tok_s", "prompt_tok_s", "wall_seconds", "runtime_total_seconds", "load_seconds"):
        values = [sample[metric] for sample in samples if sample.get(metric) is not None]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise CalibrationError("non-finite calibration metric")
        result[metric] = {"count": len(values), "median": statistics.median(values), "min": min(values),
                          "max": max(values), "population_stddev": statistics.pstdev(values)} if values else None
    median = result["generation_tok_s"]["median"] if result["generation_tok_s"] else 0
    if median <= 0:
        raise CalibrationError("generation throughput is unavailable")
    result["prediction_error_on_median_percent"] = (prediction - median) * 100 / median
    return result


def hardware_specs():
    from core.hardware import HardwareMonitor

    return HardwareMonitor().get_specs()


def run_trial(args):
    for value, low, high in ((args.repetitions, 2, 10), (args.tokens, 1, 128), (args.context, 1, 8192),
                             (args.max_seconds, 5, 300), (args.request_timeout, 1, 30)):
        if type(value) is not int or not low <= value <= high:
            raise CalibrationError("calibration budget is outside the supported range")
    base = validate_endpoint(args.url)
    deadline = time.monotonic() + args.max_seconds

    def request(path, body=None):
        return request_json(base, path, body, deadline=deadline, request_timeout=args.request_timeout)

    version = request("/api/version").get("version")
    if not isinstance(version, str) or not 0 < len(version) <= 128:
        raise CalibrationError("runtime version is unavailable")
    require_local_runtime(request("/api/status"))
    entry = model_entry(request("/api/tags"), args.model)
    show = request("/api/show", {"model": args.model})
    reject_remote(show)
    capabilities = show.get("capabilities") or []
    if not isinstance(capabilities, list) or "completion" not in capabilities:
        raise CalibrationError("model completion capability is unavailable")
    payload = {"model": args.model, "prompt": PROMPT, "stream": False, "keep_alive": "30s",
               "options": {"num_predict": args.tokens, "num_ctx": args.context, "temperature": 0, "seed": 0}}
    if "thinking" in capabilities:
        thinking = show.get("thinking") or {}
        if not isinstance(thinking, dict) or not any(value is False for value in thinking.get("values", [])):
            raise CalibrationError("thinking cannot be verifiably disabled for this bounded workload")
        payload["think"] = False
    from core import scoring
    from core.model_facts import ollama_model_facts
    from core.utils import calculate_fit

    facts = ollama_model_facts(show, entry, version)
    supported = facts["declared_supported_context"]
    if supported and args.context > supported:
        raise CalibrationError("requested context exceeds declared support")
    specs = hardware_specs()
    size_gib = entry["size"] / 1024**3
    _fit, markup_mode, _resource = calculate_fit(size_gib, specs)
    mode = re.sub(r"\[[^]]*\]", "", markup_mode)
    prediction = scoring.estimate_tok_per_s(size_gib, specs["gpu_name"], mode, specs["backend"])
    samples = []
    warmup = None
    for index in range(args.repetitions + 1):
        start = time.monotonic()
        response = request("/api/generate", payload)
        if response.get("model") != args.model:
            raise CalibrationError("runtime generation model identity changed")
        sample = sample_metrics(response, wall_seconds=time.monotonic() - start, max_tokens=args.tokens)
        if index == 0:
            warmup = sample
        else:
            samples.append(sample)
    require_local_runtime(request("/api/status"))
    after = model_entry(request("/api/tags"), args.model)
    if after["digest"] != entry["digest"] or request("/api/version").get("version") != version:
        raise CalibrationError("runtime/model identity changed during calibration")
    hardware_keys = ("cpu_name", "cpu_cores", "ram_total", "ram_free", "gpu_name", "gpu_vendor",
                     "backend", "has_gpu", "vram_total", "vram_free", "gpu_count")
    return {"schema_version": 1, "kind": "runtime_calibration", "status": "measured",
            "runtime": {"name": "ollama", "version": version, "cloud_disabled": True},
            "python": sys.version.split()[0], "hardware": {key: specs.get(key) for key in hardware_keys},
            "model_facts": facts,
            "workload": {"name": "hash-explanation-v1", "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                         "requested_context": args.context, "max_generated_tokens": args.tokens,
                         "repetitions": args.repetitions, "warmup_repetitions": 1, "keep_alive_seconds": 30},
            "prediction": {"generation_tok_s": prediction, "mode": mode, "mode_is_estimated": True,
                           "size_input": "runtime disk-size proxy", "formula": "existing bandwidth heuristic",
                           "scoring_source_sha256": hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest()},
            "warmup": warmup, "samples": samples, "summary": summarize_samples(samples, prediction=prediction),
            "limits": ["Requested context/offload is not measured allocation", "Single sequential workload, not universal accuracy",
                       "Prompt rate uses reported uncached counts when available", "No quality benchmark or model download"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            try:
                print(json.dumps(run_trial(args), allow_nan=False))
            except (OSError, ValueError, TypeError) as exc:
                reason = str(exc) if isinstance(exc, CalibrationError) else "Local runtime is unavailable or returned an unsupported response."
                print(json.dumps({"status": "failed", "reason": reason}))
                raise SystemExit(1) from exc
            return
        if args.output.exists():
            raise CalibrationError("calibration output must be fresh")
        if type(args.max_seconds) is not int or not 5 <= args.max_seconds <= 300:
            raise CalibrationError("calibration wall-time budget is invalid")
        validate_endpoint(args.url)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", *sys.argv[1:]],
                                capture_output=True, text=True, timeout=args.max_seconds + 2)
        report = json.loads(result.stdout)
        if result.returncode or report.get("status") != "measured":
            raise CalibrationError(report.get("reason", "Local calibration did not complete."))
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("[calibration] completed; report saved")
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        reason = str(exc) if isinstance(exc, CalibrationError) else "Local calibration failed or exceeded its deadline; no result was saved."
        raise SystemExit("Calibration failed: " + reason) from exc


if __name__ == "__main__":
    main()
