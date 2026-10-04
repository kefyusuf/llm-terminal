"""Versioned additive CLI exports with literal data and offline comparison."""

import json

from click.testing import CliRunner

import cli


def test_search_json_preserves_partial_results_errors_and_identity(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "HardwareMonitor", lambda: SimpleNamespace(get_specs=lambda: {}))
    model = {"id": "owner/repo", "name": "literal [model]", "source": "Hugging Face",
             "target_file": "one.gguf", "resolved_revision": "a" * 40,
             "score_composite": 72, "score_provenance": {"kind": "heuristic"}}
    monkeypatch.setattr(cli, "_search_hf_models", lambda *a, **kw: ([model], ["upstream timeout"]))
    result = CliRunner().invoke(cli.cli, ["search", "model", "--provider", "huggingface", "--json"],
                                terminal_width=20)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1 and data["kind"] == "search"
    assert data["errors"] == ["upstream timeout"]
    assert data["models"][0]["id"] == "owner/repo"
    assert data["models"][0]["target_file"] == "one.gguf"
    assert data["models"][0]["resolved_revision"] == "a" * 40
    assert data["models"][0]["scores"]["composite"] == 72
    assert data["models"][0]["score_provenance"] == {"kind": "heuristic"}
    assert data["models"][0]["name"] == "literal [model]"
    assert "Warning: upstream timeout" in result.stderr


def test_hardware_plan_json_keeps_estimates_and_unknown_identity_explicit():
    result = CliRunner().invoke(cli.cli, ["plan", "model-7b", "--context", "8192", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1 and data["kind"] == "hardware_plan"
    assert data["requested_context"] == 8192
    assert data["artifact_metadata"] is None
    assert data["estimate_provenance"]["kind"] == "heuristic"
    assert "supported context" in data["estimate_provenance"]["limitation"]
    assert data["plans"]


def test_comparison_uses_saved_export_order_without_hardware_or_network(tmp_path, monkeypatch):
    def unexpected():
        raise AssertionError("offline comparison must not rediscover hardware")

    monkeypatch.setattr(cli, "HardwareMonitor", unexpected)
    path = tmp_path / "search.json"
    data = {"schema_version": 1, "kind": "search", "errors": ["partial search"], "models": [
        {"id": "one", "name": "one", "source": "Hugging Face", "scores": {"composite": 1},
         "artifact_metadata_status": "invalid"},
        {"id": "two", "name": "two", "source": "Ollama", "scores": {"composite": 2}}]}
    path.write_text(json.dumps(data), encoding="utf-8")
    result = CliRunner().invoke(cli.cli, ["compare", "--input", str(path), "two", "one", "--json"])
    assert result.exit_code == 0, result.output
    exported = json.loads(result.stdout)
    assert exported["kind"] == "comparison"
    assert [model["id"] for model in exported["models"]] == ["two", "one"]
    assert exported["models"][0]["scores"]["composite"] == 2
    assert exported["errors"] == ["partial search"]
    assert exported["models"][1]["artifact_metadata_status"] == "invalid"


def test_comparison_rejects_unsupported_schema_and_ambiguous_identity(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"schema_version": 2, "kind": "search", "models": []}', encoding="utf-8")
    result = CliRunner().invoke(cli.cli, ["compare", "--input", str(path), "one", "two", "--json"])
    assert result.exit_code != 0
    assert "schema" in result.output.lower()
    path.write_text(json.dumps({"schema_version": 1, "kind": "search", "models": [
        {"id": "one"}, {"id": "one"}, {"id": "two"}]}), encoding="utf-8")
    result = CliRunner().invoke(cli.cli, ["compare", "--input", str(path), "one", "two", "--json"])
    assert result.exit_code != 0
    assert "ambiguous" in result.output.lower()


def test_export_does_not_emit_nonfinite_json(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "HardwareMonitor", lambda: SimpleNamespace(get_specs=lambda: {}))
    monkeypatch.setattr(cli, "_search_hf_models", lambda *a, **kw: (
        [{"id": "bad", "score_composite": float("nan")}], []))
    result = CliRunner().invoke(cli.cli, ["search", "bad", "--provider", "huggingface", "--json"])
    assert result.exit_code != 0
    assert "NaN" not in result.stdout
    assert "non-finite" in result.output


def test_plan_rejects_nonpositive_requested_context():
    result = CliRunner().invoke(cli.cli, ["plan", "model-7b", "--context", "0", "--json"])
    assert result.exit_code == 2


def test_comparison_rejects_invalid_score_objects_with_diagnostic(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "search", "models": [
        {"id": "one", "scores": "invalid"}, {"id": "two", "scores": {}}]}), encoding="utf-8")
    result = CliRunner().invoke(cli.cli, ["compare", "--input", str(path), "one", "two", "--json"])
    assert result.exit_code != 0
    assert "schema" in result.output
