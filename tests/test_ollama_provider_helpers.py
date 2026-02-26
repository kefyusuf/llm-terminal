from providers.ollama_provider import (
    _extract_models_table_rows,
    _parse_size_gb,
    _select_preferred_model_variant,
)


def test_parse_size_gb_supports_gb_and_mb():
    assert _parse_size_gb("52GB") == 52.0
    parsed = _parse_size_gb("512MB")
    assert parsed is not None
    assert round(parsed, 3) == 0.5


def test_extract_models_table_rows_parses_name_and_size():
    html = """
    <html><body>
      <table>
        <thead><tr><th>Name</th><th>Size</th><th>Context</th></tr></thead>
        <tbody>
          <tr><td>qwen3-coder-next:latest</td><td>52GB</td><td>256K</td></tr>
          <tr><td>qwen3-coder-next:q8_0</td><td>85GB</td><td>256K</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    rows = _extract_models_table_rows(html)
    assert len(rows) == 2
    assert rows[0]["name"] == "qwen3-coder-next:latest"
    assert rows[0]["size_gb"] == 52.0


def test_extract_models_rows_from_card_links_when_table_missing():
    html = """
    <html><body>
      <a href="/library/qwen3-coder-next:latest">
        <p>qwen3-coder-next:latest</p>
        <p>52GB · 256K context window · Text</p>
      </a>
      <a href="/library/qwen3-coder-next:q8_0">
        <p>qwen3-coder-next:q8_0</p>
        <p>85GB · 256K context window · Text</p>
      </a>
    </body></html>
    """
    rows = _extract_models_table_rows(html, model_name="qwen3-coder-next")
    assert len(rows) == 2
    assert rows[0]["name"] == "qwen3-coder-next:latest"
    assert rows[0]["size_gb"] == 52.0


def test_select_preferred_model_variant_prefers_latest():
    rows = [
        {"name": "qwen3-coder-next:q8_0", "size_gb": 85.0},
        {"name": "qwen3-coder-next:latest", "size_gb": 52.0},
    ]
    chosen = _select_preferred_model_variant("qwen3-coder-next", rows)
    assert chosen is not None
    assert chosen["name"] == "qwen3-coder-next:latest"
