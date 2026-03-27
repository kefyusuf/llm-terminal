"""Tests for plan mode — reverse hardware analysis."""

from model_intelligence import _gpu_class_for_vram, plan_hardware_for_model


class TestPlanHardwareForModel:
    def test_returns_list_of_dicts(self):
        plans = plan_hardware_for_model("llama-3-8b")
        assert isinstance(plans, list)
        assert len(plans) > 0
        assert all(isinstance(p, dict) for p in plans)

    def test_each_plan_has_required_keys(self):
        plans = plan_hardware_for_model("llama-3-8b")
        required_keys = {
            "quant",
            "size_gb",
            "vram_needed",
            "total_mem_needed",
            "mode",
            "min_gpu_class",
            "quality_rank",
        }
        for p in plans:
            assert required_keys.issubset(set(p.keys())), (
                f"Missing keys: {required_keys - set(p.keys())}"
            )

    def test_plans_sorted_by_quality(self):
        plans = plan_hardware_for_model("llama-3-8b")
        ranks = [p["quality_rank"] for p in plans]
        assert ranks == sorted(ranks, reverse=True)

    def test_larger_model_needs_more_vram(self):
        small_plans = plan_hardware_for_model("gemma-2b")
        large_plans = plan_hardware_for_model("llama-3-70b")

        # Find Q4_K_M for both
        small_q4 = next((p for p in small_plans if p["quant"] == "Q4_K_M"), None)
        large_q4 = next((p for p in large_plans if p["quant"] == "Q4_K_M"), None)

        assert small_q4 is not None
        assert large_q4 is not None
        assert large_q4["vram_needed"] > small_q4["vram_needed"]

    def test_custom_context_length(self):
        plans_4k = plan_hardware_for_model("llama-3-8b", target_context=4096)
        plans_32k = plan_hardware_for_model("llama-3-8b", target_context=32768)

        q4_4k = next((p for p in plans_4k if p["quant"] == "Q4_K_M"), None)
        q4_32k = next((p for p in plans_32k if p["quant"] == "Q4_K_M"), None)

        assert q4_4k is not None
        assert q4_32k is not None
        assert q4_32k["size_gb"] >= q4_4k["size_gb"]

    def test_all_modes_are_valid(self):
        plans = plan_hardware_for_model("llama-3-8b")
        valid_modes = {"GPU", "GPU+CPU", "CPU"}
        for p in plans:
            assert p["mode"] in valid_modes


class TestGpuClassForVram:
    def test_small_vram(self):
        result = _gpu_class_for_vram(4.0)
        assert "4GB" in result

    def test_medium_vram(self):
        result = _gpu_class_for_vram(12.0)
        assert "12GB" in result or "16GB" in result

    def test_large_vram(self):
        result = _gpu_class_for_vram(24.0)
        assert "24GB" in result

    def test_very_large_vram(self):
        result = _gpu_class_for_vram(100.0)
        assert "Very large" in result
