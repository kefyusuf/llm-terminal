"""Declared facts, explicit assumptions and estimates stay distinct."""
import json

import pytest
from click.testing import CliRunner


def snapshot():
    from core.model_facts import ollama_model_facts

    return ollama_model_facts({"model_info": {"general.architecture": "llama",
                "llama.context_length": 8192, "llama.block_count": 32,
                "llama.embedding_length": 4096, "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8},
                "details": {"quantization_level": "Q4_K_M"}, "capabilities": ["completion"]},
                {"name": "fixture:latest", "digest": "a" * 64, "size": 4 * 1024**3}, "fixture-runtime")


def test_metadata_kv_math_scales_cache_but_shares_weight_proxy():
    from core.model_facts import memory_scenario

    result = memory_scenario(snapshot(), requested_context=4096, concurrency=2, gpu_weight_percent=50)
    assert result["status"] == "estimated"
    assert result["components"]["kv_cache_bytes"] == 1024**3
    assert result["components"]["weight_proxy_bytes"] == 4 * 1024**3
    assert result["estimated_gpu_bytes"] == 3 * 1024**3 + 256 * 1024**2
    assert result["estimated_cpu_weight_bytes"] == 2 * 1024**3
    assert result["assumptions"]["runtime_allocation_measured"] is False


def test_declared_context_does_not_become_requested_context_support():
    from core.model_facts import memory_scenario

    result = memory_scenario(snapshot(), requested_context=16384)
    assert result["status"] == "context_exceeds_declared_support"
    assert result["estimated_gpu_bytes"] is None


@pytest.mark.parametrize("architecture", [None, "gemma4", "mamba"])
def test_unsupported_cache_architectures_are_not_given_llama_estimates(architecture):
    from core.model_facts import memory_scenario

    facts = snapshot()
    facts["architecture"] = architecture
    assert memory_scenario(facts, requested_context=4096)["status"] == "unsupported_architecture"


def test_missing_or_boolean_dimensions_are_unknown():
    from core.model_facts import memory_scenario, ollama_model_facts

    facts = ollama_model_facts({"model_info": {"general.architecture": "llama", "llama.block_count": True}},
                               {"name": "fixture:latest", "digest": "a" * 64, "size": 100}, "test")
    assert facts["layers"] is None
    assert memory_scenario(facts, requested_context=4096)["status"] == "insufficient_metadata"


def test_sliding_window_cache_is_not_silently_treated_as_full_attention():
    from core.model_facts import memory_scenario

    facts = snapshot()
    facts["special_cache_layout"] = True
    assert memory_scenario(facts, requested_context=4096)["status"] == "unsupported_cache_layout"


@pytest.mark.parametrize("options", [{"concurrency": True}, {"gpu_weight_percent": 101}, {"kv_type": "q4"}])
def test_invalid_assumptions_are_rejected(options):
    from core.model_facts import memory_scenario

    with pytest.raises(ValueError, match="assumption"):
        memory_scenario(snapshot(), requested_context=4096, **options)


def test_saved_facts_cli_is_additive_and_does_not_rescore(tmp_path, monkeypatch):
    import cli

    facts = snapshot()
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts))
    legacy = [{"quant": "existing", "quality_rank": 1}]
    monkeypatch.setattr("core.model_intelligence.plan_hardware_for_model", lambda *a, **kw: legacy)
    result = CliRunner().invoke(cli.cli, ["plan", "fixture:latest", "--facts", str(path), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["plans"] == legacy
    assert data["metadata_plan"]["status"] == "estimated"
    assert data["metadata_plan"]["facts_provenance"]["revalidated"] is False
    wrong = CliRunner().invoke(cli.cli, ["plan", "other:latest", "--facts", str(path), "--json"])
    assert wrong.exit_code != 0
    assert "identity" in wrong.output.lower()


def test_invalid_explicit_head_dimension_does_not_become_a_derived_fact():
    from core.model_facts import ollama_model_facts

    facts = ollama_model_facts({"model_info": {"general.architecture": "llama",
          "llama.embedding_length": 4096, "llama.attention.head_count": 32,
          "llama.attention.key_length": True}}, {"name": "fixture"}, "test")
    assert facts["key_head_dim"] is None
    assert facts["value_head_dim"] == 128


def test_unknown_calibration_envelope_schema_is_rejected(tmp_path):
    from core.model_facts import read_model_facts

    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"schema_version": 2, "kind": "runtime_calibration", "model_facts": snapshot()}))
    with pytest.raises(ValueError, match="schema"):
        read_model_facts(path, "fixture:latest")


def test_plain_saved_facts_plan_explains_allocation_limit(tmp_path):
    import cli

    path = tmp_path / "facts.json"
    path.write_text(json.dumps(snapshot()))
    result = CliRunner().invoke(cli.cli, ["plan", "fixture:latest", "--facts", str(path)])
    assert result.exit_code == 0, result.output
    assert "runtime allocation was not measured" in result.output
