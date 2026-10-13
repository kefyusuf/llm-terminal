"""Optional installed-model lookup must not multiply the search wait budget."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.mark.parametrize("status, expected", [(500, []), (200, ["deepseek", "qwen"])])
def test_installed_model_probe_uses_one_attempt_and_keeps_name_contract(monkeypatch, status, expected):
    import config
    from providers.ollama_provider import get_installed_ollama_models

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            body = json.dumps({"models": [{"name": "DeepSeek:7b"}, {"name": "Qwen:0.5b"}]}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(config.settings, "ollama_api_base", f"http://127.0.0.1:{server.server_port}")
    try:
        assert get_installed_ollama_models() == expected
        assert calls == ["/api/tags"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
