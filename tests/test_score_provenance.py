"""Contracts for explaining scores without changing numeric ranking."""

import pytest

from core.scoring import enrich_result_with_scores, score_model


def scored(**overrides):
    args = {
        "model_name": "example",
        "size_gb": 4.8,
        "params": "8B",
        "quant": "Q4_K_M",
        "use_case_key": "general",
        "specs": {"gpu_name": "RTX 4090", "vram_total": 24, "ram_total": 32},
        "mode": "GPU",
    }
    args.update(overrides)
    return score_model(**args)


def test_scores_disclose_heuristics_and_context_is_not_capacity():
    result = scored()
    provenance = result.provenance
    assert provenance["schema_version"] == 1
    assert provenance["kind"] == "heuristic"
    assert provenance["measured"] is False
    assert provenance["quality"]["basis"] == "parameters_and_quantization"
    assert provenance["context"]["basis"] == "model_size_proxy"
    assert "not a context-window measurement" in provenance["context"]["limitation"]
    assert provenance["fit"]["memory_basis"] == "total_capacity"
    assert provenance["composite"]["weights"]["quality"] == 0.3
    assert (result.quality, result.context, result.estimated_tok_s) == (62, 45, 115.5)


@pytest.mark.parametrize(
    "specs,source,bandwidth",
    [
        ({"gpu_name": "RTX 4090"}, "gpu_lookup", 1008),
        ({"gpu_name": "unknown", "backend": "rocm"}, "backend_default", 180),
        ({"gpu_name": "", "backend": "unknown"}, "cuda_default", 220),
    ],
)
def test_speed_provenance_matches_actual_fallback(specs, source, bandwidth):
    speed = scored(specs=specs).provenance["speed"]
    assert speed["bandwidth_source"] == source
    assert speed["bandwidth_gb_s"] == bandwidth
    assert speed["efficiency"] == 0.55


def test_no_fit_discloses_suppressed_throughput():
    result = scored(mode="No Fit")
    assert result.estimated_tok_s == 0
    assert result.provenance["speed"]["computed"] is False


def test_unknown_inputs_disclose_fallbacks_and_general_weights():
    provenance = scored(params="-", quant="GGUF", use_case_key="unknown", mode="-").provenance
    assert provenance["quality"]["quantization_default"] is True
    assert provenance["speed"]["mode_default"] is True
    assert provenance["composite"]["use_case"] == "general"
    assert provenance["composite"]["use_case_default"] is True


def test_enrichment_preserves_scores_and_exposes_size_input_source():
    result = enrich_result_with_scores(
        {"name": "example", "size": "~4.8 GB", "params": "8B", "quant": "Q4_K_M", "mode": "GPU"}, {}
    )
    assert result["score_provenance"]["size_source"] == "display_size_parse"
    assert result["score_provenance"]["size_gb"] == 4.8
    assert result["score_context"] == 45
    enrich_result_with_scores(result, {})
    assert result["score_provenance"]["size_source"] == "display_size_parse"
    result["_size_gb"] = 10
    enrich_result_with_scores(result, {})
    assert result["score_provenance"]["size_source"] == "supplied_size_gb"
    assert result["score_context"] == 65
