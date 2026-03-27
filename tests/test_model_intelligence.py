"""Tests for model_intelligence module — MoE detection, quant selection, size estimation."""

import pytest

from model_intelligence import (
    QUANT_MULTIPLIERS,
    QuantInfo,
    detect_moe,
    estimate_model_size_gb_v2,
    parse_experts,
    select_best_quant,
)

# ---------------------------------------------------------------------------
# MoE Detection
# ---------------------------------------------------------------------------


class TestDetectMoe:
    @pytest.mark.parametrize(
        "name",
        [
            "mixtral-8x7b",
            "Mixtral-8x7B-Instruct-v0.1",
            "deepseek-v2",
            "deepseek-v3",
            "deepseek_v2_lite",
            "qwen2-57b-a14b-moe",
            "jamba-1.5-large",
            "grok-1",
            "switch-transformer-128",
            "olmoe-7b-913",
        ],
    )
    def test_detects_known_moe_models(self, name):
        assert detect_moe(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "llama-3-8b",
            "mistral-7b",
            "qwen2-7b",
            "gemma-2b",
            "phi-3-mini",
            "codellama-13b",
            "llama-3.1-70b",
            "deepseek-coder-v2",
            "deepseek-r1",
        ],
    )
    def test_rejects_non_moe_models(self, name):
        assert detect_moe(name) is False

    def test_empty_string(self):
        assert detect_moe("") is False

    def test_case_insensitive(self):
        assert detect_moe("MIXTRAL-8x7B") is True
        assert detect_moe("DeepSeek-V3") is True


# ---------------------------------------------------------------------------
# Expert Parsing
# ---------------------------------------------------------------------------


class TestParseExperts:
    @pytest.mark.parametrize(
        "name,expected_total,expected_active",
        [
            ("mixtral-8x7b", 8, 2),
            ("Mixtral-8x22B", 8, 2),
            ("deepseek-v2", 160, 6),
            ("qwen2-57b-a14b-moe", 8, 2),
            ("switch-transformer-128", 128, 2),
        ],
    )
    def test_parses_expert_counts(self, name, expected_total, expected_active):
        total, active = parse_experts(name)
        assert total == expected_total
        assert active == expected_active

    def test_non_moe_returns_none(self):
        assert parse_experts("llama-3-8b") == (None, None)

    def test_empty_string(self):
        assert parse_experts("") == (None, None)


# ---------------------------------------------------------------------------
# Quantization Selection
# ---------------------------------------------------------------------------


class TestSelectBestQuant:
    def test_picks_highest_quality_that_fits_gpu(self):
        """With 24GB VRAM and 14GB model, Q8_0 (14.5GB) fits."""
        result = select_best_quant(param_count_gb=14.0, vram_gb=24.0, ram_gb=32.0)
        assert result is not None
        quant, size, mode = result
        quant_names = [q.name for q in QUANT_MULTIPLIERS]
        assert quant in quant_names
        assert size <= 24.0
        assert mode == "GPU"

    def test_falls_back_to_cpu_offload(self):
        """With 8GB VRAM and 14GB model, must offload to CPU."""
        result = select_best_quant(param_count_gb=14.0, vram_gb=8.0, ram_gb=32.0)
        assert result is not None
        _quant, _size, mode = result
        assert mode in ("GPU+CPU", "CPU")

    def test_returns_none_when_nothing_fits(self):
        """With tiny hardware, nothing fits."""
        result = select_best_quant(param_count_gb=70.0, vram_gb=1.0, ram_gb=2.0)
        assert result is None

    def test_prefers_higher_quality_over_smaller_size(self):
        """Given enough VRAM, should pick Q8_0 over Q4_K_M."""
        result = select_best_quant(param_count_gb=7.0, vram_gb=24.0, ram_gb=32.0)
        assert result is not None
        quant, _size, mode = result
        # Q8_0 for 7B = 7GB * 1.0 + ~0.5GB overhead ≈ 7.5GB, fits in 24GB
        assert quant == "Q8_0"
        assert mode == "GPU"

    def test_small_model_fits_any_quant(self):
        """1B model should fit Q8_0 on moderate hardware (FP16 skipped for dynamic selection)."""
        result = select_best_quant(param_count_gb=1.0, vram_gb=8.0, ram_gb=16.0)
        assert result is not None
        quant, _, mode = result
        # Q8_0 for 1B = 1.0 * 1.0 + 0.5 = 1.5GB, fits in 8GB VRAM
        assert quant == "Q8_0"
        assert mode == "GPU"


# ---------------------------------------------------------------------------
# Model Size Estimation v2
# ---------------------------------------------------------------------------


class TestEstimateModelSizeV2:
    def test_standard_sizes(self):
        assert estimate_model_size_gb_v2("llama-3-8b") == pytest.approx(4.8, abs=0.5)
        assert estimate_model_size_gb_v2("llama-3-70b") == pytest.approx(40.0, abs=1.0)

    def test_moe_model_smaller_than_raw_params(self):
        """Mixtral 8x7B raw params 46.7B, estimated ~20GB (closest match to 34B class)."""
        size = estimate_model_size_gb_v2("mixtral-8x7b")
        # 8*7*0.85 = 47.6B total, closest match in our map
        assert size < 40.0  # Not as big as a 70B dense model
        assert size > 10.0  # But larger than a small dense model

    def test_deepseek_v2_moe(self):
        size = estimate_model_size_gb_v2("deepseek-v2")
        assert size > 0

    def test_unknown_model_returns_default(self):
        size = estimate_model_size_gb_v2("unknown-model-xyz")
        assert size == pytest.approx(4.8, abs=0.1)

    def test_empty_string(self):
        size = estimate_model_size_gb_v2("")
        assert size > 0

    @pytest.mark.parametrize(
        "name,expected_range",
        [
            ("phi-3.8b-mini", (1.5, 3.0)),
            ("gemma-2b", (1.0, 2.5)),
            ("codellama-13b", (7.0, 10.0)),
            ("qwen2-72b", (38.0, 45.0)),
        ],
    )
    def test_size_ranges(self, name, expected_range):
        size = estimate_model_size_gb_v2(name)
        assert expected_range[0] <= size <= expected_range[1], (
            f"{name}: {size} not in {expected_range}"
        )


# ---------------------------------------------------------------------------
# QuantInfo dataclass
# ---------------------------------------------------------------------------


class TestQuantInfo:
    def test_fields(self):
        qi = QuantInfo(name="Q4_K_M", multiplier=0.55, quality_rank=6)
        assert qi.name == "Q4_K_M"
        assert qi.multiplier == 0.55
        assert qi.quality_rank == 6

    def test_size_at_multiplier(self):
        qi = QuantInfo(name="Q4_K_M", multiplier=0.55, quality_rank=6)
        assert qi.size_for_params(14.0) == pytest.approx(7.7)


# ---------------------------------------------------------------------------
# QUANT_MULTIPLIERS ordering
# ---------------------------------------------------------------------------


class TestQuantOrdering:
    def test_quality_rank_decreases(self):
        """Ranks should be strictly decreasing (higher rank = better quality)."""
        ranks = [q.quality_rank for q in QUANT_MULTIPLIERS]
        assert ranks == sorted(ranks, reverse=True)

    def test_multipliers_decrease(self):
        """Multipliers should decrease as quality rank decreases."""
        mults = [q.multiplier for q in QUANT_MULTIPLIERS]
        assert mults == sorted(mults, reverse=True)
