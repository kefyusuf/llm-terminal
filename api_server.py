"""REST API server for AI Model Explorer.

Provides a machine-readable HTTP API on port 8787 for programmatic access
to model search, scoring, and hardware analysis.

Usage:
    python -m api_server          # Start on localhost:8787
    python -m api_server --port 9000  # Custom port
"""

from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from loguru import logger

import config
from core.errors import ProviderError
from core.hardware import HardwareMonitor
from core.model_intelligence import plan_hardware_for_model
from core.scoring import score_model
from core.utils import (
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    infer_quant_from_name,
)
from providers import detect_available_providers
from providers.capabilities import get_all_provider_capabilities
from providers.hf_provider import search_hf_models
from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

API_VERSION = "1.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
REST_MODEL_PROVIDER_SLUGS = ("ollama", "huggingface")
VALID_MODEL_PROVIDERS = {"all", *REST_MODEL_PROVIDER_SLUGS}
MAX_MODEL_LIMIT = 100
PROVIDER_API_BASES = {
    "huggingface": "https://huggingface.co",
    "lmstudio": "http://localhost:1234",
    "docker": "http://localhost:12434",
    "mlx": "local",
}


def get_provider_api_bases() -> dict[str, str]:
    """Return provider API bases using current runtime configuration where available."""
    return {
        **PROVIDER_API_BASES,
        "ollama": config.settings.ollama_api_base,
    }


def get_rest_model_provider_slugs() -> tuple[str, ...]:
    """Return provider slugs directly supported by the REST models endpoint."""
    return REST_MODEL_PROVIDER_SLUGS


def serialize_provider_error(error: ProviderError) -> dict:
    """Serialize one structured provider diagnostic for REST responses."""
    return {
        "provider": error.provider,
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
        "status_code": error.status_code,
        "retry_after_seconds": error.retry_after_seconds,
    }


def build_provider_descriptors(
    availability: dict[str, bool], api_bases: dict[str, str]
) -> list[dict]:
    """Build REST provider descriptors without conflating global and endpoint capabilities."""
    rest_model_providers = set(get_rest_model_provider_slugs())
    providers = []
    for slug, capabilities in get_all_provider_capabilities().items():
        providers.append(
            {
                "name": slug,
                "display_name": capabilities.display_name,
                "available": availability.get(slug, False),
                "api_base": api_bases.get(slug, ""),
                "models_endpoint": slug in rest_model_providers,
                "capabilities": {
                    "searchable": capabilities.searchable,
                    "detectable": capabilities.detectable,
                    "lists_installed": capabilities.lists_installed,
                    "downloadable": capabilities.downloadable,
                    "paginated": capabilities.paginated,
                },
            }
        )
    return providers


def smoke_mode_enabled() -> bool:
    """Return whether API smoke mode is enabled for the current process."""
    return os.getenv("AIMODEL_SMOKE") == "1"


class ModelAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the model API."""

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
            logger.warning("API request failed: {}", exc)
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
        """Handle model search requests across supported REST providers."""
        query = params.get("search", [""])[0]
        provider = params.get("provider", ["all"])[0].lower()
        if provider not in VALID_MODEL_PROVIDERS:
            return self._error(
                f"Invalid 'provider' parameter '{provider}'; expected one of: "
                f"{', '.join(sorted(VALID_MODEL_PROVIDERS))}.",
                400,
            )

        try:
            limit = int(params.get("limit", ["20"])[0])
        except (ValueError, IndexError):
            return self._error("Invalid 'limit' parameter; expected integer.", 400)
        if not 1 <= limit <= MAX_MODEL_LIMIT:
            return self._error(
                f"Invalid 'limit' parameter; expected integer between 1 and {MAX_MODEL_LIMIT}.",
                400,
            )

        min_fit = params.get("min_fit", ["all"])[0].lower()
        use_case = params.get("use_case", ["all"])[0].lower()
        sort_by = params.get("sort", ["composite"])[0].lower()

        valid_sorts = {"composite", "speed", "quality", "name"}
        if sort_by not in valid_sorts:
            return self._error(f"Invalid 'sort' parameter '{sort_by}'.", 400)

        specs = self.monitor.get_specs()
        results = []
        errors: list[str] = []
        structured_errors: list[ProviderError] = []

        if provider in ("all", "ollama"):
            local = get_installed_ollama_models()
            ollama_results, ollama_errors, _ = search_ollama_models(
                query or "*",
                specs,
                local,
                page_size=limit,
                _structured_error_sink=structured_errors.append,
            )
            results.extend(ollama_results)
            errors.extend(ollama_errors)

        if provider in ("all", "huggingface"):
            hf_results, hf_errors = search_hf_models(
                query or "*",
                specs,
                {},
                limit=limit,
                hf_token=config.settings.hf_token,
                _structured_error_sink=structured_errors.append,
            )
            results.extend(hf_results)
            errors.extend(hf_errors)

        if use_case != "all":
            results = [r for r in results if r.get("use_case_key") == use_case]
        if min_fit != "all":
            results = [r for r in results if min_fit in r.get("fit", "").lower()]

        if sort_by == "composite":
            results.sort(key=lambda r: r.get("score_composite", 0), reverse=True)
        elif sort_by == "speed":
            results.sort(key=lambda r: r.get("score_speed", 0), reverse=True)
        elif sort_by == "quality":
            results.sort(key=lambda r: r.get("score_quality", 0), reverse=True)
        elif sort_by == "name":
            results.sort(key=lambda r: r.get("name", "").lower())

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
                "errors": errors,
                "structured_errors": [
                    serialize_provider_error(error) for error in structured_errors
                ],
            }
        )

    def _handle_models_top(self, params):
        try:
            limit = int(params.get("limit", ["5"])[0])
        except (ValueError, IndexError):
            return self._error("Invalid 'limit' parameter; expected integer.", 400)

        params["sort"] = ["composite"]
        params["limit"] = [str(limit)]
        self._handle_models(params)

    def _handle_plan(self, model_name, params):
        try:
            context = int(params.get("context", ["4096"])[0])
        except (ValueError, IndexError):
            return self._error("Invalid 'context' parameter; expected integer.", 400)
        if context <= 0:
            return self._error("Invalid 'context' parameter; expected a positive integer.", 400)

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
        """Return provider availability plus global and REST-surface metadata."""
        availability = detect_available_providers()
        api_bases = get_provider_api_bases()
        providers = build_provider_descriptors(availability, api_bases)
        self._json_response(
            {
                "providers": providers,
                "models_endpoint_providers": list(get_rest_model_provider_slugs()),
            }
        )


class _SmokeHardwareMonitor:
    """Minimal monitor used only by the transport-level API health smoke."""

    def get_specs(self) -> dict:
        return {}


class _LocalThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that binds locally without reverse/FQDN resolution."""

    def server_bind(self) -> None:
        """Bind the listening socket without the stdlib HTTPServer FQDN lookup."""
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    monitor: HardwareMonitor | None = None,
) -> ThreadingHTTPServer:
    """Create and configure the API server with an injectable hardware monitor."""
    ModelAPIHandler.monitor = monitor if monitor is not None else HardwareMonitor()
    server = _LocalThreadingHTTPServer((host, port), ModelAPIHandler)
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
    *,
    monitor: HardwareMonitor | None = None,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the API server in a background thread with autonomous request threads.

    Returns ``(server, thread)`` for later shutdown via ``server.shutdown()``.
    """
    server = create_server(host, port, monitor=monitor)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="api-server")
    thread.start()
    return server, thread


def run_smoke_check() -> int:
    """Run a bounded transport-level API health check against an ephemeral server."""
    server, thread = start_server_background(DEFAULT_HOST, 0, monitor=_SmokeHardwareMonitor())
    port = int(server.server_address[1])

    try:
        with urllib.request.urlopen(f"http://{DEFAULT_HOST}:{port}/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") != "ok":
            raise SystemExit("[smoke] api health check failed")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    print("[smoke] api server ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run API smoke mode or start the configured REST server."""
    args = list(sys.argv[1:] if argv is None else argv)
    if smoke_mode_enabled():
        return run_smoke_check()

    port = DEFAULT_PORT
    if len(args) > 1 and args[0] == "--port":
        port = int(args[1])
    run_server(port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
