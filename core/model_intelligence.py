"""Model intelligence module — MoE detection, dynamic quantization, and size estimation.

Provides tools for understanding model architecture (MoE vs dense), selecting
optimal quantization levels for given hardware, and estimating model sizes
with awareness of expert-offloading patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# MoE Detection
# ---------------------------------------------------------------------------

MOE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"mixtral", re.IGNORECASE),
    re.compile(r"deepseek[_-]v[23]", re.IGNORECASE),
    re.compile(r"qwen.*moe", re.IGNORECASE),
    re.compile(r"jamba", re.IGNORECASE),
    re.compile(r"grok", re.IGNORECASE),
    re.compile(r"switch", re.IGNORECASE),
    re.compile(r"nllb.*moe", re.IGNORECASE),
    re.compile(r"olmoe", re.IGNORECASE),
]


def detect_moe(name: str) -> bool:
    """Return True if *name* matches a known Mixture-of-Experts model pattern."""
    if not name:
        return False
    return any(p.search(name) for p in MOE_PATTERNS)


# ---------------------------------------------------------------------------
# Expert Parsing
# ---------------------------------------------------------------------------

# Pattern: NxM where N = number of experts, M = params per expert
_EXPERT_PATTERN = re.compile(r"(\d+)x(\d+)b?", re.IGNORECASE)
# Pattern: "a{N}b" for active params (e.g., qwen2-57b-a14b-moe)
_ACTIVE_PATTERN = re.compile(r"a(\d+)b", re.IGNORECASE)
# Pattern: "-N" standalone count after model type (e.g., switch-transformer-128)
_STANDALONE_EXPERT_PATTERN = re.compile(r"-(\d+)(?:b|$)", re.IGNORECASE)
# Known large MoE models with non-standard naming
_LARGE_MOE_EXPERTS: dict[str, tuple[int, int]] = {
    "deepseek-v2": (160, 6),
    "deepseek-v3": (256, 8),
}
# Models where standalone number is expert count (not param count)
_STANDALONE_EXPERT_MODELS = ["switch"]


def parse_experts(name: str) -> tuple[int | None, int | None]:
    """Extract (total_experts, active_experts) from a model name.

    Returns ``(None, None)`` when the model is not MoE or experts cannot
    be parsed.
    """
    if not detect_moe(name):
        return None, None

    lower = name.lower()

    # Check known large MoE models first
    for key, (total, active) in _LARGE_MOE_EXPERTS.items():
        if key in lower:
            return total, active

    # Parse NxM pattern (e.g., "8x7b", "8x22b")
    match = _EXPERT_PATTERN.search(name)
    if match:
        total = int(match.group(1))
        active = 2  # Most MoE models activate 2 experts per token
        return total, active

    # Parse "a{N}b" active-param pattern (e.g., qwen2-57b-a14b-moe)
    active_match = _ACTIVE_PATTERN.search(name)
    if active_match:
        # Estimate total experts from active params (typical ratio: 1:4 to 1:8)
        # Default to 8 experts with 2 active
        return 8, 2

    # Parse standalone expert count for known model families (e.g., switch-transformer-128)
    for model_prefix in _STANDALONE_EXPERT_MODELS:
        if model_prefix in lower:
            standalone = _STANDALONE_EXPERT_PATTERN.search(name)
            if standalone:
                total = int(standalone.group(1))
                return total, 2

    return None, None


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantInfo:
    """Information about a single quantization level."""

    name: str
    multiplier: float  # Size multiplier relative to FP16 params
    quality_rank: int  # Higher = better quality (FP16 = 10)

    def size_for_params(self, param_count_gb: float) -> float:
        """Return estimated model size in GB at this quantization level."""
        return param_count_gb * self.multiplier


# Ordered from highest quality to lowest
QUANT_MULTIPLIERS: list[QuantInfo] = [
    QuantInfo("FP16", 2.0, 10),
    QuantInfo("Q8_0", 1.0, 9),
    QuantInfo("Q6_K", 0.75, 8),
    QuantInfo("Q5_K_M", 0.65, 7),
    QuantInfo("Q5_0", 0.625, 6),
    QuantInfo("Q4_K_M", 0.55, 5),
    QuantInfo("Q4_0", 0.5, 4),
    QuantInfo("Q3_K", 0.4, 3),
    QuantInfo("Q2_K", 0.3, 2),
]

# Context overhead estimate (KV cache, activations) in GB
CONTEXT_OVERHEAD_GB = 0.5


def select_best_quant(
    param_count_gb: float,
    vram_gb: float,
    ram_gb: float,
    context_overhead: float = CONTEXT_OVERHEAD_GB,
) -> tuple[str, float, str] | None:
    """Select the highest-quality quantization that fits the available hardware.

    Tries quantizations from Q8_0 (best quality) down to Q2_K (smallest).
    First tries GPU-only, then GPU+CPU offload, then CPU-only.

    Returns:
        ``(quant_name, total_size_gb, mode)`` where mode is ``"GPU"``,
        ``"GPU+CPU"``, or ``"CPU"``. Returns ``None`` if nothing fits.
    """
    for qi in QUANT_MULTIPLIERS:
        if qi.name == "FP16":
            continue  # Skip FP16 for dynamic selection (usually too large)

        size = qi.size_for_params(param_count_gb) + context_overhead

        if size <= vram_gb:
            return qi.name, size, "GPU"
        if size <= vram_gb + ram_gb:
            return qi.name, size, "GPU+CPU"
        if size <= ram_gb:
            return qi.name, size, "CPU"

    return None


# ---------------------------------------------------------------------------
# Size Estimation
# ---------------------------------------------------------------------------

# Standard param-count to size mapping (Q4_K_M baseline, ~0.55x params)
_PARAM_SIZE_MAP: dict[str, float] = {
    "0.2b": 0.2,
    "0.3b": 0.2,
    "0.5b": 0.6,
    "0.6b": 0.5,
    "0.8b": 0.5,
    "1b": 0.8,
    "1.5b": 1.2,
    "2b": 1.4,
    "3b": 2.0,
    "3.8b": 2.2,
    "7b": 4.8,
    "8b": 4.8,
    "13b": 8.0,
    "14b": 8.0,
    "27b": 16.0,
    "32b": 19.0,
    "34b": 20.0,
    "47b": 26.0,
    "57b": 32.0,
    "70b": 40.0,
    "72b": 40.0,
    "180b": 100.0,
    "405b": 230.0,
}


def _extract_param_token(name: str) -> str | None:
    """Extract parameter count token from model name (e.g., '7b', '70b')."""
    match = re.search(r"(\d+(?:\.\d+)?[Bb])", name)
    return match.group(1).lower() if match else None


def _find_closest_param_key(total_params: float) -> str | None:
    """Find the closest param-size key for a given total parameter count (in billions)."""
    best_key = None
    best_diff = float("inf")
    for key, _size in _PARAM_SIZE_MAP.items():
        key_b = float(key.replace("b", ""))
        diff = abs(key_b - total_params)
        if diff < best_diff:
            best_diff = diff
            best_key = key
    return best_key


def estimate_model_size_gb_v2(name: str) -> float:
    """Estimate model size in GB with MoE awareness.

    For MoE models, calculates the full model size (all experts stored in memory).
    The active expert ratio affects speed, not memory footprint — all weights
    must be resident even if only a subset are active per token.
    """
    if not name:
        return 4.8  # Default fallback

    # MoE-aware estimation
    if detect_moe(name):
        total_experts, active_experts = parse_experts(name)
        if total_experts is not None and active_experts is not None:
            # Check for NxM pattern to get total param count
            match = _EXPERT_PATTERN.search(name)
            if match:
                experts_n = int(match.group(1))
                params_per_expert = float(match.group(2))
                # Total params: experts * per-expert * ~0.85 (shared layers are smaller)
                total_params = experts_n * params_per_expert * 0.85
                # Find closest param size key
                closest_key = _find_closest_param_key(total_params)
                if closest_key:
                    return _PARAM_SIZE_MAP[closest_key]

            # Fallback: use named param count for MoE models
            param_token = _extract_param_token(name)
            if param_token and param_token in _PARAM_SIZE_MAP:
                return _PARAM_SIZE_MAP[param_token]

    # Dense model — use standard mapping
    param_token = _extract_param_token(name)
    if param_token and param_token in _PARAM_SIZE_MAP:
        return _PARAM_SIZE_MAP[param_token]

    # Fallback: try substring matching
    mn = name.lower()
    for key in sorted(_PARAM_SIZE_MAP.keys(), key=len, reverse=True):
        if key in mn:
            return _PARAM_SIZE_MAP[key]

    return 4.8


def estimate_context_overhead_gb(context_length: int = 4096) -> float:
    """Estimate KV cache overhead in GB for a given context length.

    Rough heuristic: ~1MB per 1K context tokens, scaled.
    """
    return (context_length / 1024) * 0.001 + CONTEXT_OVERHEAD_GB


# ---------------------------------------------------------------------------
# Plan Mode — Reverse Hardware Analysis
# ---------------------------------------------------------------------------


def plan_hardware_for_model(
    model_name: str,
    target_context: int = 4096,
) -> list[dict]:
    """Compute hardware requirements for a model across quantization levels.

    Returns a list of dicts, one per viable quantization, each containing:
    - ``quant``: quantization name
    - ``size_gb``: estimated model size
    - ``vram_needed``: minimum VRAM for GPU-only inference
    - ``total_mem_needed``: minimum total memory (VRAM+RAM) for offload
    - ``mode``: recommended inference mode (GPU, GPU+CPU, CPU)
    - ``min_gpu_class``: human-readable minimum GPU recommendation
    """
    size_gb = estimate_model_size_gb_v2(model_name)
    context_overhead = estimate_context_overhead_gb(target_context)
    plans = []

    for qi in QUANT_MULTIPLIERS:
        quant_size = qi.size_for_params(size_gb) + context_overhead
        gpu_only = quant_size
        with_offload = quant_size * 0.7  # ~70% on GPU, rest on CPU
        if qi.multiplier >= 1.0:
            mode = "GPU"
            min_mem = gpu_only
        elif qi.multiplier >= 0.5:
            mode = "GPU+CPU"
            min_mem = with_offload
        else:
            mode = "GPU+CPU"
            min_mem = with_offload

        plans.append(
            {
                "quant": qi.name,
                "size_gb": round(quant_size, 1),
                "vram_needed": round(gpu_only, 1),
                "total_mem_needed": round(min_mem, 1),
                "mode": mode,
                "min_gpu_class": _gpu_class_for_vram(gpu_only),
                "quality_rank": qi.quality_rank,
            }
        )

    return plans


_GPU_CLASSES: list[tuple[float, str]] = [
    (4, "Integrated / Low-end"),
    (8, "RTX 4060 / 3050 / Apple M2"),
    (12, "RTX 3060 / 4070"),
    (16, "RTX 4060 Ti / A4000"),
    (24, "RTX 4090 / 3090 / A5000"),
    (48, "RTX A6000 / A40"),
    (80, "A100 80GB / H100"),
]


def _gpu_class_for_vram(vram_gb: float) -> str:
    """Return a human-readable GPU class recommendation for a VRAM requirement."""
    for threshold, gpu_class in _GPU_CLASSES:
        if vram_gb <= threshold:
            return f"~{threshold}GB+ ({gpu_class})"
    return "Very large (>80GB VRAM)"
