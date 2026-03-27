"""4-dimension scoring engine for LLM model evaluation.

Scores models on Quality, Speed, Fit, and Context dimensions (0-100 each),
then computes a use-case-weighted composite score.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_intelligence import QUANT_MULTIPLIERS

# ---------------------------------------------------------------------------
# GPU Bandwidth Lookup Table (GB/s)
# ---------------------------------------------------------------------------

GPU_BANDWIDTH: dict[str, int] = {
    # NVIDIA — RTX 40 series
    "RTX 4090": 1008,
    "RTX 4080 SUPER": 717,
    "RTX 4080": 717,
    "RTX 4070 Ti SUPER": 672,
    "RTX 4070 Ti": 504,
    "RTX 4070 SUPER": 504,
    "RTX 4070": 504,
    "RTX 4060 Ti": 288,
    "RTX 4060": 272,
    # NVIDIA — RTX 30 series
    "RTX 3090 Ti": 1008,
    "RTX 3090": 936,
    "RTX 3080 Ti": 912,
    "RTX 3080": 760,
    "RTX 3070 Ti": 608,
    "RTX 3070": 448,
    "RTX 3060 Ti": 448,
    "RTX 3060": 360,
    "RTX 3050": 224,
    # NVIDIA — RTX 20 series
    "RTX 2080 Ti": 616,
    "RTX 2080 SUPER": 496,
    "RTX 2080": 448,
    "RTX 2070 SUPER": 448,
    "RTX 2070": 448,
    "RTX 2060": 336,
    # NVIDIA — Data Center
    "H100": 3350,
    "H200": 4800,
    "A100 80GB": 2039,
    "A100 40GB": 1555,
    "L40S": 864,
    "L40": 864,
    "L4": 300,
    "A30": 933,
    "A10": 600,
    "T4": 300,
    "V100": 900,
    "P100": 732,
    # NVIDIA — GTX
    "GTX 1080 Ti": 484,
    "GTX 1080": 320,
    "GTX 1070": 256,
    # AMD — Consumer
    "RX 7900 XTX": 960,
    "RX 7900 XT": 800,
    "RX 7900 GRE": 576,
    "RX 7800 XT": 624,
    "RX 7700 XT": 432,
    "RX 7600": 288,
    "RX 6900 XT": 512,
    "RX 6800 XT": 512,
    "RX 6800": 512,
    "RX 6700 XT": 384,
    # AMD — Data Center
    "MI300X": 5200,
    "MI250X": 3277,
    "MI250": 3277,
    "MI210": 1600,
    "MI100": 1228,
    # Apple Silicon
    "M3 Ultra": 800,
    "M3 Max": 400,
    "M3 Pro": 200,
    "M3": 100,
    "M2 Ultra": 800,
    "M2 Max": 400,
    "M2 Pro": 200,
    "M2": 100,
    "M1 Ultra": 800,
    "M1 Max": 400,
    "M1 Pro": 200,
    "M1": 68,
    # Intel Arc
    "Arc A770": 560,
    "Arc A750": 512,
    "Arc A380": 192,
}

# Backend fallback bandwidths (GB/s) when GPU is unknown
_BACKEND_DEFAULTS: dict[str, int] = {
    "cuda": 220,
    "rocm": 180,
    "metal": 160,
    "sycl": 140,
    "cpu": 50,
}

# Efficiency multipliers per inference mode
_MODE_EFFICIENCY: dict[str, float] = {
    "GPU": 0.55,
    "GPU+CPU": 0.30,
    "CPU": 0.18,
    "No Fit": 0.0,
}


# ---------------------------------------------------------------------------
# Use-Case Weights
# ---------------------------------------------------------------------------

USE_CASE_WEIGHTS: dict[str, dict[str, float]] = {
    "chat": {"quality": 0.25, "speed": 0.35, "fit": 0.25, "context": 0.15},
    "coding": {"quality": 0.35, "speed": 0.30, "fit": 0.20, "context": 0.15},
    "reasoning": {"quality": 0.55, "speed": 0.15, "fit": 0.15, "context": 0.15},
    "vision": {"quality": 0.30, "speed": 0.25, "fit": 0.25, "context": 0.20},
    "math": {"quality": 0.45, "speed": 0.20, "fit": 0.20, "context": 0.15},
    "embedding": {"quality": 0.30, "speed": 0.40, "fit": 0.20, "context": 0.10},
    "general": {"quality": 0.30, "speed": 0.25, "fit": 0.25, "context": 0.20},
}


# ---------------------------------------------------------------------------
# Scores Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    """Four-dimension model scores and weighted composite."""

    quality: int
    speed: int
    fit: int
    context: int
    composite: int
    estimated_tok_s: float


# ---------------------------------------------------------------------------
# GPU Bandwidth Lookup
# ---------------------------------------------------------------------------


def find_gpu_bandwidth(gpu_name: str) -> int | None:
    """Find the memory bandwidth in GB/s for a GPU by name.

    Does case-insensitive substring matching against the lookup table.
    Returns ``None`` if no match is found.
    """
    if not gpu_name:
        return None

    lower = gpu_name.lower()

    # Direct substring match (case-insensitive)
    for key, bw in GPU_BANDWIDTH.items():
        if key.lower() in lower:
            return bw

    # Try matching just the model number/name part
    for key, bw in GPU_BANDWIDTH.items():
        key_parts = key.lower().split()
        for part in key_parts:
            if len(part) > 2 and part in lower:
                return bw

    return None


# ---------------------------------------------------------------------------
# Speed Estimation
# ---------------------------------------------------------------------------


def estimate_tok_per_s(
    model_size_gb: float,
    gpu_name: str,
    mode: str,
    backend: str = "cuda",
) -> float:
    """Estimate tokens per second for a model on given hardware.

    Formula: ``tok/s = (bandwidth_GB/s / model_size_GB) x efficiency_factor``

    Args:
        model_size_gb: Model size in GB.
        gpu_name: GPU name string for bandwidth lookup.
        mode: Inference mode (``"GPU"``, ``"GPU+CPU"``, ``"CPU"``, ``"No Fit"``).
        backend: Backend type for fallback bandwidth (``"cuda"``, ``"rocm"``,
                 ``"metal"``, etc.).

    Returns:
        Estimated tokens per second, or ``0.0`` for No Fit.
    """
    if mode == "No Fit" or model_size_gb <= 0:
        return 0.0

    # Get bandwidth
    bw = find_gpu_bandwidth(gpu_name)
    if bw is None:
        bw = _BACKEND_DEFAULTS.get(backend, _BACKEND_DEFAULTS["cuda"])

    # Get efficiency
    efficiency = _MODE_EFFICIENCY.get(mode, _MODE_EFFICIENCY["GPU"])

    # Calculate
    tok_s = (bw / model_size_gb) * efficiency
    return round(tok_s, 1)


# ---------------------------------------------------------------------------
# Dimension Score Computation
# ---------------------------------------------------------------------------


def compute_quality_score(params: str, quant: str) -> int:
    """Compute quality score (0-100) based on parameter count and quantization.

    Larger models with less aggressive quantization score higher.
    """
    # Parameter count score (0-70 points)
    param_score = 0
    if params and params != "-":
        try:
            param_str = params.upper().replace("B", "").replace("M", "")
            param_val = float(param_str)
            if "M" in params.upper():
                param_val /= 1000  # Convert millions to billions
            # Log scale: 1B→20, 7B→40, 13B→50, 70B→65, 405B→70
            import math

            param_score = min(70, int(15 * math.log2(param_val + 1)))
        except (ValueError, TypeError):
            param_score = 15  # Unknown params get a base score

    # Quantization quality score (0-30 points)
    quant_score = 15  # Default
    quant_upper = quant.upper() if quant else ""
    for qi in QUANT_MULTIPLIERS:
        if qi.name in quant_upper:
            # Map quality_rank (2-10) to (5-30) points
            quant_score = int((qi.quality_rank / 10) * 30)
            break

    return min(100, param_score + quant_score)


def compute_fit_score(
    model_size_gb: float,
    vram_gb: float,
    ram_gb: float,
    mode: str,
) -> int:
    """Compute fit score (0-100) based on VRAM utilization efficiency.

    Sweet spot is 50-80% VRAM utilization. Too tight (<10% free) or too
    wasteful (>95% free) scores lower.
    """
    if mode == "No Fit":
        return 0
    if mode == "CPU":
        return max(5, 30 - int(model_size_gb / max(ram_gb, 1) * 100))

    if mode == "GPU+CPU":
        total_mem = vram_gb + ram_gb
        utilization = model_size_gb / total_mem if total_mem > 0 else 1.0
        base = 40
    else:  # GPU
        utilization = model_size_gb / vram_gb if vram_gb > 0 else 1.0
        base = 60

    # Sweet spot: 50-80% utilization
    if 0.5 <= utilization <= 0.8:
        return min(100, base + 30)
    elif utilization < 0.5:
        # Under-utilized: penalize waste
        waste_penalty = int((0.5 - utilization) * 60)
        return max(10, base + 20 - waste_penalty)
    else:
        # Over-utilized: penalize tight fit
        tight_penalty = int((utilization - 0.8) * 200)
        return max(5, base + 20 - tight_penalty)


def compute_context_score(size_gb: float) -> int:
    """Compute context score (0-100) based on model size.

    Larger models typically support larger context windows. This is a
    heuristic — ideally we'd use actual context window data.
    """
    # Models under 2B: limited context (score 20-40)
    # Models 2B-13B: moderate context (score 40-60)
    # Models 13B-70B: good context (score 60-80)
    # Models 70B+: excellent context (score 80-100)
    if size_gb < 1.5:
        return 25
    elif size_gb < 5:
        return 45
    elif size_gb < 10:
        return 55
    elif size_gb < 22:
        return 65
    elif size_gb < 45:
        return 75
    else:
        return min(100, 80 + int(size_gb / 50))


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------


def score_model(
    model_name: str,
    size_gb: float,
    params: str,
    quant: str,
    use_case_key: str,
    specs: dict,
    mode: str,
) -> Scores:
    """Compute all four dimension scores and weighted composite for a model.

    Args:
        model_name: Model name (unused currently, reserved for future).
        size_gb: Estimated model size in GB.
        params: Parameter count string (e.g., "8B", "70B").
        quant: Quantization level string.
        use_case_key: Use-case category key for weight selection.
        specs: Hardware specs dict from HardwareMonitor.
        mode: Inference mode ("GPU", "GPU+CPU", "CPU", "No Fit").

    Returns:
        ``Scores`` dataclass with all dimension and composite values.
    """
    # Dimension scores
    quality = compute_quality_score(params, quant)
    fit = compute_fit_score(
        size_gb,
        specs.get("vram_total", 0),
        specs.get("ram_total", 0),
        mode,
    )
    context = compute_context_score(size_gb)

    # Speed: requires GPU name
    gpu_name = specs.get("gpu_name", "")
    backend = specs.get("backend", "cuda")
    estimated_tok_s = estimate_tok_per_s(size_gb, gpu_name, mode, backend)

    # Map tok/s to 0-100 score
    # 0 tok/s → 0, 20 tok/s → 30, 50 tok/s → 50, 100 tok/s → 70, 200+ → 90+
    if estimated_tok_s <= 0:
        speed = 0
    elif estimated_tok_s < 10:
        speed = int(estimated_tok_s * 3)
    elif estimated_tok_s < 50:
        speed = 30 + int((estimated_tok_s - 10) * 0.5)
    elif estimated_tok_s < 200:
        speed = 50 + int((estimated_tok_s - 50) * 0.133)
    else:
        speed = min(100, 70 + int((estimated_tok_s - 200) * 0.05))

    # Composite: weighted average
    weights = USE_CASE_WEIGHTS.get(use_case_key, USE_CASE_WEIGHTS["general"])
    composite = round(
        quality * weights["quality"]
        + speed * weights["speed"]
        + fit * weights["fit"]
        + context * weights["context"]
    )

    return Scores(
        quality=quality,
        speed=speed,
        fit=fit,
        context=context,
        composite=composite,
        estimated_tok_s=estimated_tok_s,
    )


def enrich_result_with_scores(result: dict, specs: dict) -> dict:
    """Add 4-dimension scoring fields to a model result dict in-place.

    Reads ``size_gb`` (float), ``params``, ``quant``, ``use_case_key``,
    and ``mode`` from *result*, computes scores, and writes
    ``score_quality``, ``score_speed``, ``score_fit``, ``score_context``,
    ``score_composite``, and ``estimated_tok_s`` back into *result*.

    Also adds MoE metadata from model_intelligence if applicable.
    Returns the mutated *result* dict for convenience.
    """
    from model_intelligence import detect_moe, parse_experts

    name = result.get("name", "")
    params = result.get("params", "-")
    quant = result.get("quant", "GGUF")
    use_case_key = result.get("use_case_key", "general")
    mode_text = _strip_mode(result.get("mode", "-"))
    size_gb = result.get("_size_gb", 0.0)

    # If we don't have a float size, try to parse from display string
    if not size_gb:
        size_text = result.get("size", "0")
        size_gb = _parse_size_text(size_text)
        result["_size_gb"] = size_gb

    scores = score_model(
        model_name=name,
        size_gb=size_gb,
        params=params,
        quant=quant,
        use_case_key=use_case_key,
        specs=specs,
        mode=mode_text,
    )

    result["score_quality"] = scores.quality
    result["score_speed"] = scores.speed
    result["score_fit"] = scores.fit
    result["score_context"] = scores.context
    result["score_composite"] = scores.composite
    result["estimated_tok_s"] = scores.estimated_tok_s

    # MoE metadata
    result["is_moe"] = detect_moe(name)
    if result["is_moe"]:
        total, active = parse_experts(name)
        result["total_experts"] = total or 0
        result["active_experts"] = active or 0
    else:
        result["total_experts"] = 0
        result["active_experts"] = 0

    return result


def _strip_mode(mode_text: str) -> str:
    """Extract plain mode string from Rich markup (e.g. '[green]GPU[/green]' -> 'GPU')."""
    import re

    clean = re.sub(r"\[[^\]]+\]", "", mode_text).strip()
    if not clean or clean == "-":
        return "No Fit"
    if "gpu" in clean.lower() and "cpu" in clean.lower():
        return "GPU+CPU"
    if "gpu" in clean.lower():
        return "GPU"
    if "cpu" in clean.lower():
        return "CPU"
    return "No Fit"


def _parse_size_text(size_text: str) -> float:
    """Parse a size string like '4.8 GB' or '~4.8 GB' into float GB."""
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*GB", size_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match_mb = re.search(r"(\d+(?:\.\d+)?)\s*MB", size_text, re.IGNORECASE)
    if match_mb:
        return float(match_mb.group(1)) / 1024
    return 0.0
