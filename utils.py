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
    match = re.search(r"(\d+(?:\.\d+)?[BbMm])", name, re.IGNORECASE)
    return match.group(1).upper() if match else "-"


def format_likes(num):
    if num >= 1000000:
        return f"{num / 1000000:.1f}M"
    if num >= 1000:
        return f"{num / 1000:.1f}K"
    return str(num)


def determine_use_case_key(name):
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
    return USE_CASE_LABELS[determine_use_case_key(name)]


def calculate_fit(size_gb, specs):
    buffer = 1.5
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
    mn = model_name.lower()
    if "70b" in mn or "72b" in mn:
        return 40.0
    if "32b" in mn:
        return 19.0
    if "27b" in mn:
        return 16.0
    if "14b" in mn or "13b" in mn:
        return 8.0
    if "8b" in mn or "7b" in mn:
        return 4.8
    if "1.5b" in mn:
        return 1.2
    if "0.5b" in mn:
        return 0.6
    return 4.8
