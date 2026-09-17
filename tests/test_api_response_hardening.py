from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from unittest.mock import patch

from api_server import DEFAULT_HOST, ModelAPIHandler, create_server


@contextmanager
def _running_server():
    server = create_server(DEFAULT_HOST, 0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_api_does_not_enable_cross_origin_browser_access_by_default():
    with (
        _running_server() as port,
        urllib.request.urlopen(f"http://{DEFAULT_HOST}:{port}/health", timeout=5) as response,
    ):
        assert response.headers.get("Access-Control-Allow-Origin") is None


def test_unexpected_api_exception_returns_generic_500_without_internal_detail():
    secret = "/Users/private/project/.env: API_TOKEN=should-not-leak"

    with (
        patch.object(ModelAPIHandler, "_handle_health", side_effect=RuntimeError(secret)),
        _running_server() as port,
    ):
        request = urllib.request.Request(f"http://{DEFAULT_HOST}:{port}/health")
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            payload = json.loads(body)
            assert exc.code == 500
            assert payload == {"error": "Internal server error."}
            assert secret not in body
        else:  # pragma: no cover - the RED/contract path must be an HTTP 500
            raise AssertionError("unexpected API exception did not produce HTTP 500")
