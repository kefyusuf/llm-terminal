from providers.hf_provider import _format_hf_http_error
from providers.ollama_provider import _retry_after_from_response


class _DummyResponse:
    def __init__(self, status_code, headers):
        self.status_code = status_code
        self.headers = headers


class _DummyHFError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__("hf error")
        self.response = _DummyResponse(status_code, headers or {})


def test_hf_rate_limit_message_includes_retry_after():
    message = _format_hf_http_error(_DummyHFError(429, {"Retry-After": "42"}))
    assert "rate-limited" in message
    assert "42s" in message


def test_ollama_retry_after_parser_handles_invalid_values():
    response = _DummyResponse(429, {"Retry-After": "invalid"})
    assert _retry_after_from_response(response) is None
