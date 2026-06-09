import re

USE_CASE_LABELS = {
    "coding": "[bold blue]Coding[/bold blue]",
    "vision": "[bold magenta]Vision[/bold magenta]",
    "math": "[bold cyan]Math[/bold cyan]",
    "reasoning": "[bold yellow]Reasoning[/bold yellow]",
    "embedding": "[grey74]Embedding[/grey74]",
    "chat": "[bold green]Chat[/bold green]",
    "general": "[white]General[/white]",
}


def extract_params(name):
    """Extract the parameter-count token (e.g. ``"8B"``) from a model name.

    Returns ``"-"`` when no match is found.
    """
    match = re.search(r"(\d+(?:\.\d+)?[BbMm])", name, re.IGNORECASE)
    return match.group(1).upper() if match else "-"


def format_likes(num):
    """Format a raw like count into a compact human-readable string (e.g. ``"4.2K"``)."""
    if num >= 1000000:
        return f"{num / 1000000:.1f}M"
    if num >= 1000:
        return f"{num / 1000:.1f}K"
    return str(num)


def determine_use_case_key(name):
    """Return the plain use-case category key for *name* (e.g. ``"coding"``)."""
    name_lower = name.lower()
    if any(kw in name_lower for kw in ["coder", "code", "starcoder", "deepseek-coder"]):
        return "coding"
    if any(kw in name_lower for kw in ["vision", "vl", "llava", "pixtral"]):
        return "vision"
    if any(kw in name_lower for kw in ["math"]):
        return "math"
    if any(kw in name_lower for kw in ["reasoning", "think", "-r1", "deepseek-r1"]):
        return "reasoning"
    if any(kw in name_lower for kw in ["embed", "bge", "nomic"]):
        return "embedding"
    if any(kw in name_lower for kw in ["instruct", "chat", "dolphin", "hermes"]):
        return "chat"
    return "general"


def determine_use_case(name):
    """Return the Rich-markup use-case label for *name* (e.g. ``"[bold blue]Coding[/bold blue]"``)."""
    return USE_CASE_LABELS[determine_use_case_key(name)]


def calculate_fit(size_gb, specs):
    """Estimate whether a model of *size_gb* fits in the available hardware.

    Args:
        size_gb: Estimated model size in gigabytes.
        specs: Hardware specification dict from :class:`~core.hardware.HardwareMonitor`.

    Returns:
        A 3-tuple ``(fit_markup, mode_markup, resource_label)`` — all Rich markup
        strings suitable for direct use in Textual widgets.
    """
    buffer = 1.0  # Reduced from 1.5GB for more accurate fit
    req = size_gb + buffer
    if specs["has_gpu"] and specs["vram_free"] >= req:
        return "[bold green]Perfect[/bold green]", "[green]GPU[/green]", "VRAM"
    if specs["has_gpu"] and (specs["vram_free"] + specs["ram_free"]) >= req:
        return (
            "[bold yellow]Partial[/bold yellow]",
            "[yellow]GPU+CPU[/yellow]",
            "VRAM+RAM",
        )
    if specs["ram_free"] >= req:
        return "[bold yellow]Slow[/bold yellow]", "[yellow]CPU[/yellow]", "RAM"
    return "[bold red]No Fit[/bold red]", "[red]-[/red]", "Insufficient"


def estimate_model_size_gb(model_name):
    """Estimate model disk-size in GB using the MoE-aware v2 estimator.

    Delegates to :func:`core.model_intelligence.estimate_model_size_gb_v2`.
    """
    from core.model_intelligence import estimate_model_size_gb as _v2

    return _v2(model_name)


def infer_quant_from_name(name, default="GGUF"):
    """Infer the GGUF quantisation level from a model filename or identifier.

    Returns *default* when no recognised quantisation token is found.
    """
    nm = name.lower().replace("-", "_")
    if "q4_k_m" in nm:
        return "Q4_K_M"
    if "q5_k_m" in nm:
        return "Q5_K_M"
    if "q8_0" in nm:
        return "Q8_0"
    if "q6_k" in nm:
        return "Q6_K"
    if "q5_0" in nm:
        return "Q5_0"
    if "q5_1" in nm:
        return "Q5_1"
    if "q4_0" in nm:
        return "Q4_0"
    if "q4_1" in nm:
        return "Q4_1"
    if "q3_k" in nm:
        return "Q3_K"
    if "q2_k" in nm:
        return "Q2_K"
    if "fp16" in nm:
        return "FP16"
    return default


# ---------------------------------------------------------------------------
# Shared parsing helpers (used by multiple modules)
# ---------------------------------------------------------------------------


def parse_retry_after_seconds(header_value):
    """Parse a ``Retry-After`` HTTP header value into an integer number of seconds.

    Args:
        header_value: Raw header string (e.g. ``"30"``), or ``None``.

    Returns:
        Seconds as an ``int``, or ``None`` if the value is absent or unparseable.
    """
    if not header_value:
        return None
    try:
        return int(header_value)
    except (TypeError, ValueError):
        return None


def extract_download_progress(line):
    """Extract a percentage value from a single line of download output.

    Args:
        line: A text line such as ``"Pulling layer … 47%"``.

    Returns:
        Integer percentage in the range ``[0, 100]``, or ``None`` if not found.
    """
    if not line:
        return None
    match = re.search(r"(\d{1,3})%", line)
    if not match:
        return None
    value = int(match.group(1))
    if 0 <= value <= 100:
        return value
    return None
