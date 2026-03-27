"""REST API server for AI Model Explorer.

Provides a machine-readable HTTP API on port 8787 for programmatic access
to model search, scoring, and hardware analysis.

Usage:
    python api_server.py          # Start on localhost:8787
    python api_server.py --port 9000  # Custom port
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware import HardwareMonitor  # noqa: E402
from model_intelligence import plan_hardware_for_model  # noqa: E402
from providers.hf_provider import search_hf_models  # noqa: E402
from providers.ollama_provider import get_installed_ollama_models, search_ollama_models  # noqa: E402
from scoring import score_model  # noqa: E402
from utils import (  # noqa: E402
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    infer_quant_from_name,
)

API_VERSION = "1.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


class ModelAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the model API."""

    # Shared state (set by server startup)
    monitor: HardwareMonitor = None  # type: ignore[assignment]

    def log_message(self, format, *args):
        """Suppress default request logging to stderr."""
        pass

    def _json_response(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=400):
        self._json_response({"error": message}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        try:
            if path == "/health":
                self._handle_health()
            elif path == "/api/v1/system":
                self._handle_system()
            elif path == "/api/v1/models":
                self._handle_models(params)
            elif path == "/api/v1/models/top":
                self._handle_models_top(params)
            elif path.startswith("/api/v1/models/") and path.endswith("/plan"):
                model_name = path.split("/api/v1/models/")[1].replace("/plan", "")
                self._handle_plan(model_name, params)
            elif path.startswith("/api/v1/scores/"):
                model_name = path.split("/api/v1/scores/")[1]
                self._handle_scores(model_name)
            elif path == "/api/v1/providers":
                self._handle_providers()
            else:
                self._error(f"Unknown endpoint: {path}", 404)
        except Exception as exc:
            self._error(str(exc), 500)

    def _handle_health(self):
        self._json_response(
            {
                "status": "ok",
                "api_version": API_VERSION,
                "service": "ai-model-explorer-api",
            }
        )

    def _handle_system(self):
        specs = self.monitor.get_specs()
        self._json_response(
            {
                "cpu_name": specs.get("cpu_name", ""),
                "cpu_cores": specs.get("cpu_cores", 0),
                "ram_total_gb": round(specs.get("ram_total", 0), 1),
                "ram_free_gb": round(specs.get("ram_free", 0), 1),
                "gpu_name": specs.get("gpu_name", ""),
                "gpu_vendor": specs.get("gpu_vendor"),
                "backend": specs.get("backend", "cpu"),
                "gpu_count": specs.get("gpu_count", 0),
                "vram_total_gb": round(specs.get("vram_total", 0), 1),
                "vram_free_gb": round(specs.get("vram_free", 0), 1),
                "has_gpu": specs.get("has_gpu", False),
            }
        )

    def _handle_models(self, params):
        query = params.get("search", [""])[0]
        provider = params.get("provider", ["all"])[0].lower()
        limit = int(params.get("limit", ["20"])[0])
        min_fit = params.get("min_fit", ["all"])[0].lower()
        use_case = params.get("use_case", ["all"])[0].lower()
        sort_by = params.get("sort", ["composite"])[0].lower()

        specs = self.monitor.get_specs()
        results = []

        # Search providers
        if provider in ("all", "ollama"):
            local = get_installed_ollama_models()
            ollama_results, _, _ = search_ollama_models(query or "*", specs, local, page_size=limit)
            results.extend(ollama_results)

        if provider in ("all", "huggingface"):
            hf_results, _ = search_hf_models(query or "*", specs, {}, limit=limit)
            results.extend(hf_results)

        # Filter
        if use_case != "all":
            results = [r for r in results if r.get("use_case_key") == use_case]
        if min_fit != "all":
            results = [r for r in results if min_fit in r.get("fit", "").lower()]

        # Sort
        if sort_by == "composite":
            results.sort(key=lambda r: r.get("score_composite", 0), reverse=True)
        elif sort_by == "speed":
            results.sort(key=lambda r: r.get("score_speed", 0), reverse=True)
        elif sort_by == "quality":
            results.sort(key=lambda r: r.get("score_quality", 0), reverse=True)
        elif sort_by == "name":
            results.sort(key=lambda r: r.get("name", "").lower())

        # Serialize
        models = []
        for r in results[:limit]:
            models.append(
                {
                    "name": r.get("name", ""),
                    "source": r.get("source", ""),
                    "publisher": r.get("publisher", ""),
                    "params": r.get("params", "-"),
                    "quant": r.get("quant", ""),
                    "size": r.get("size", ""),
                    "use_case": r.get("use_case_key", "general"),
                    "fit": r.get("fit", ""),
                    "mode": r.get("mode", ""),
                    "scores": {
                        "quality": r.get("score_quality", 0),
                        "speed": r.get("score_speed", 0),
                        "fit": r.get("score_fit", 0),
                        "context": r.get("score_context", 0),
                        "composite": r.get("score_composite", 0),
                        "estimated_tok_s": r.get("estimated_tok_s", 0),
                    },
                    "moe": {
                        "is_moe": r.get("is_moe", False),
                        "total_experts": r.get("total_experts", 0),
                        "active_experts": r.get("active_experts", 0),
                    },
                }
            )

        self._json_response(
            {
                "models": models,
                "total": len(models),
                "query": query,
                "provider": provider,
            }
        )

    def _handle_models_top(self, params):
        limit = int(params.get("limit", ["5"])[0])

        # Forward to models endpoint with composite sort
        params["sort"] = ["composite"]
        params["limit"] = [str(limit)]
        self._handle_models(params)

    def _handle_plan(self, model_name, params):
        context = int(params.get("context", ["4096"])[0])
        plans = plan_hardware_for_model(model_name, target_context=context)
        self._json_response(
            {
                "model": model_name,
                "context_length": context,
                "plans": plans,
            }
        )

    def _handle_scores(self, model_name):
        specs = self.monitor.get_specs()
        size_gb = estimate_model_size_gb(model_name)
        params_str = extract_params(model_name)
        quant = infer_quant_from_name(model_name)
        use_case_key = determine_use_case_key(model_name)

        scores = score_model(
            model_name=model_name,
            size_gb=size_gb,
            params=params_str,
            quant=quant,
            use_case_key=use_case_key,
            specs=specs,
            mode="GPU" if specs.get("has_gpu") else "CPU",
        )

        self._json_response(
            {
                "model": model_name,
                "scores": {
                    "quality": scores.quality,
                    "speed": scores.speed,
                    "fit": scores.fit,
                    "context": scores.context,
                    "composite": scores.composite,
                    "estimated_tok_s": scores.estimated_tok_s,
                },
                "size_gb": size_gb,
                "params": params_str,
                "quant": quant,
                "use_case": use_case_key,
            }
        )

    def _handle_providers(self):
        providers = []
        # Check Ollama
        from hardware import check_ollama_running

        providers.append(
            {
                "name": "ollama",
                "available": check_ollama_running(),
                "api_base": "http://localhost:11434",
            }
        )
        # HuggingFace always available
        providers.append(
            {
                "name": "huggingface",
                "available": True,
                "api_base": "https://huggingface.co",
            }
        )
        self._json_response({"providers": providers})


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Create and configure the API server."""
    ModelAPIHandler.monitor = HardwareMonitor()
    server = ThreadingHTTPServer((host, port), ModelAPIHandler)
    return server


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Run the API server (blocking)."""
    server = create_server(host, port)
    print(f"AI Model Explorer API v{API_VERSION}")
    print(f"Listening on http://{host}:{port}")
    print("Endpoints: /health, /api/v1/system, /api/v1/models, /api/v1/models/top, ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        server.shutdown()


def start_server_background(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the API server in a background thread.

    Returns ``(server, thread)`` for later shutdown via ``server.shutdown()``.
    """
    server = create_server(host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="api-server")
    thread.start()
    return server, thread


if __name__ == "__main__":
    port = DEFAULT_PORT
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        port = int(sys.argv[2])
    run_server(port=port)
