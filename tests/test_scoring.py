"""Tests for the 4-dimension scoring engine."""

from scoring import (
    USE_CASE_WEIGHTS,
    Scores,
    compute_fit_score,
    compute_quality_score,
    estimate_tok_per_s,
    find_gpu_bandwidth,
    score_model,
)

# ---------------------------------------------------------------------------
# GPU Bandwidth Lookup
# ---------------------------------------------------------------------------


class TestFindGpuBandwidth:
    def test_known_nvidia_gpu(self):
        bw = find_gpu_bandwidth("NVIDIA GeForce RTX 4090")
        assert bw == 1008

    def test_known_amd_gpu(self):
        bw = find_gpu_bandwidth("AMD Radeon RX 7900 XTX")
        assert bw == 960

    def test_known_apple_gpu(self):
        bw = find_gpu_bandwidth("Apple M2 Ultra")
        assert bw == 800

    def test_partial_match(self):
        bw = find_gpu_bandwidth("GeForce RTX 3080 12GB")
        assert bw == 760

    def test_unknown_gpu_returns_none(self):
        bw = find_gpu_bandwidth("Some Random GPU 3000")
        assert bw is None

    def test_empty_string(self):
        bw = find_gpu_bandwidth("")
        assert bw is None


# ---------------------------------------------------------------------------
# Speed Estimation
# ---------------------------------------------------------------------------


class TestEstimateTokPerS:
    def test_high_end_gpu(self):
        # RTX 4090: 1008 GB/s, 8B model @ Q4 ≈ 4.8GB
        tok_s = estimate_tok_per_s(
            model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 4090", mode="GPU"
        )
        assert tok_s > 50  # Should be fast
        assert tok_s < 500

    def test_cpu_offload_slower(self):
        gpu_tok = estimate_tok_per_s(
            model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 4090", mode="GPU"
        )
        offload_tok = estimate_tok_per_s(
            model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 4090", mode="GPU+CPU"
        )
        assert offload_tok < gpu_tok

    def test_cpu_only_slowest(self):
        gpu_tok = estimate_tok_per_s(
            model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 4090", mode="GPU"
        )
        cpu_tok = estimate_tok_per_s(
            model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 4090", mode="CPU"
        )
        assert cpu_tok < gpu_tok

    def test_larger_model_slower(self):
        small_tok = estimate_tok_per_s(
            model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 3090", mode="GPU"
        )
        large_tok = estimate_tok_per_s(
            model_size_gb=40.0, gpu_name="NVIDIA GeForce RTX 3090", mode="GPU"
        )
        assert large_tok < small_tok

    def test_unknown_gpu_uses_backend_default(self):
        tok_s = estimate_tok_per_s(model_size_gb=4.8, gpu_name="Unknown GPU", mode="GPU")
        assert tok_s > 0

    def test_no_fit_returns_zero(self):
        tok_s = estimate_tok_per_s(
            model_size_gb=4.8, gpu_name="NVIDIA GeForce RTX 4090", mode="No Fit"
        )
        assert tok_s == 0.0


# ---------------------------------------------------------------------------
# Quality Score
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    def test_large_model_scores_higher(self):
        s70b = compute_quality_score(params="70B", quant="Q4_K_M")
        s7b = compute_quality_score(params="7B", quant="Q4_K_M")
        assert s70b > s7b

    def test_better_quant_scores_higher(self):
        q8 = compute_quality_score(params="7B", quant="Q8_0")
        q2 = compute_quality_score(params="7B", quant="Q2_K")
        assert q8 > q2

    def test_score_bounds(self):
        score = compute_quality_score(params="7B", quant="Q4_K_M")
        assert 0 <= score <= 100

    def test_very_large_model(self):
        score = compute_quality_score(params="405B", quant="Q8_0")
        assert score >= 90

    def test_unknown_params(self):
        score = compute_quality_score(params="-", quant="Q4_K_M")
        assert score > 0  # Should still have some base score
        assert score < 50


# ---------------------------------------------------------------------------
# Fit Score
# ---------------------------------------------------------------------------


class TestComputeFitScore:
    def test_perfect_fit_scores_high(self):
        # Model uses ~50% of VRAM — sweet spot
        score = compute_fit_score(model_size_gb=12.0, vram_gb=24.0, ram_gb=32.0, mode="GPU")
        assert score >= 80

    def test_too_tight_scores_low(self):
        # Model uses >95% of VRAM
        score = compute_fit_score(model_size_gb=23.0, vram_gb=24.0, ram_gb=32.0, mode="GPU")
        assert score < 60

    def test_too_small_scores_moderate(self):
        # Model uses only 5% of VRAM — wasteful
        score = compute_fit_score(model_size_gb=1.0, vram_gb=24.0, ram_gb=32.0, mode="GPU")
        assert score < 80

    def test_cpu_offload_scores_lower(self):
        gpu_score = compute_fit_score(model_size_gb=12.0, vram_gb=24.0, ram_gb=32.0, mode="GPU")
        offload_score = compute_fit_score(
            model_size_gb=12.0, vram_gb=6.0, ram_gb=32.0, mode="GPU+CPU"
        )
        assert offload_score < gpu_score

    def test_no_fit_scores_zero(self):
        score = compute_fit_score(model_size_gb=100.0, vram_gb=4.0, ram_gb=8.0, mode="No Fit")
        assert score == 0


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------


class TestScoreModel:
    def test_returns_scores_object(self):
        scores = score_model(
            model_name="llama-3-8b",
            size_gb=4.8,
            params="8B",
            quant="Q4_K_M",
            use_case_key="chat",
            specs={
                "vram_total": 24.0,
                "vram_free": 20.0,
                "ram_total": 32.0,
                "ram_free": 28.0,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "has_gpu": True,
            },
            mode="GPU",
        )
        assert isinstance(scores, Scores)
        assert 0 <= scores.quality <= 100
        assert 0 <= scores.speed <= 100
        assert 0 <= scores.fit <= 100
        assert 0 <= scores.context <= 100
        assert 0 <= scores.composite <= 100

    def test_composite_uses_use_case_weights(self):
        chat_scores = score_model(
            model_name="llama-3-8b",
            size_gb=4.8,
            params="8B",
            quant="Q4_K_M",
            use_case_key="chat",
            specs={
                "vram_total": 24,
                "vram_free": 20,
                "ram_total": 32,
                "ram_free": 28,
                "gpu_name": "RTX 4090",
                "has_gpu": True,
            },
            mode="GPU",
        )
        reasoning_scores = score_model(
            model_name="llama-3-8b",
            size_gb=4.8,
            params="8B",
            quant="Q4_K_M",
            use_case_key="reasoning",
            specs={
                "vram_total": 24,
                "vram_free": 20,
                "ram_total": 32,
                "ram_free": 28,
                "gpu_name": "RTX 4090",
                "has_gpu": True,
            },
            mode="GPU",
        )
        # Chat weights speed higher, reasoning weights quality higher
        # Same model, different weights → different composite scores
        assert chat_scores.composite != reasoning_scores.composite

    def test_unknown_use_case_falls_back_to_general(self):
        scores = score_model(
            model_name="llama-3-8b",
            size_gb=4.8,
            params="8B",
            quant="Q4_K_M",
            use_case_key="unknown_category",
            specs={
                "vram_total": 24,
                "vram_free": 20,
                "ram_total": 32,
                "ram_free": 28,
                "gpu_name": "RTX 4090",
                "has_gpu": True,
            },
            mode="GPU",
        )
        assert scores.composite > 0


# ---------------------------------------------------------------------------
# Context Score
# ---------------------------------------------------------------------------


class TestContextScore:
    def test_larger_models_get_higher_context_score(self):
        """Larger models typically support larger context windows."""
        small = score_model(
            model_name="phi-2",
            size_gb=1.5,
            params="3B",
            quant="Q4_K_M",
            use_case_key="general",
            specs={
                "vram_total": 24,
                "vram_free": 20,
                "ram_total": 32,
                "ram_free": 28,
                "gpu_name": "RTX 4090",
                "has_gpu": True,
            },
            mode="GPU",
        )
        large = score_model(
            model_name="llama-3-70b",
            size_gb=40.0,
            params="70B",
            quant="Q4_K_M",
            use_case_key="general",
            specs={
                "vram_total": 80,
                "vram_free": 70,
                "ram_total": 128,
                "ram_free": 100,
                "gpu_name": "A100",
                "has_gpu": True,
            },
            mode="GPU",
        )
        assert large.context >= small.context


# ---------------------------------------------------------------------------
# USE_CASE_WEIGHTS
# ---------------------------------------------------------------------------


class TestUseCaseWeights:
    def test_all_weights_sum_to_one(self):
        for case, weights in USE_CASE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"Weights for '{case}' sum to {total}"

    def test_all_cases_have_all_dimensions(self):
        dims = {"quality", "speed", "fit", "context"}
        for _case, weights in USE_CASE_WEIGHTS.items():
            assert set(weights.keys()) == dims, "'case' missing dimensions"
