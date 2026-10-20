"""Search-related constants extracted from AIModelViewer."""

RESULTS_COLUMN_LABELS_COMFORTABLE = {
    "inst": "Install",
    "source": "Source",
    "publisher": "Provider",
    "name": "Model",
    "params": "Scale",
    "use_case": "Use Case",
    "score": "Score",
    "quant": "Format",
    "mode": "Mode",
    "fit": "Fit",
    "download": "Download",
}

RESULTS_COLUMN_LABELS_COMPACT = {
    "inst": "In",
    "source": "Src",
    "publisher": "Prov",
    "name": "Model",
    "params": "Param",
    "use_case": "Use",
    "score": "Score",
    "quant": "Quant",
    "mode": "Mode",
    "fit": "Fit",
    "download": "D/L",
}

USE_CASE_OPTIONS = [
    ("all", "Any Use"),
    ("chat", "Chat"),
    ("coding", "Coding"),
    ("vision", "Vision"),
    ("reasoning", "Reason"),
    ("math", "Math"),
    ("embedding", "Embed"),
    ("general", "General"),
]

SORT_OPTIONS = [
    ("score", "Score"),
    ("downloads", "Downloads"),
    ("name", "Name"),
]

FIT_OPTIONS = [
    ("all", "All"),
    ("fit", "Fit"),
    ("partial", "Partial"),
    ("nofit", "No Fit"),
]

USE_CASE_COMPACT_TAGS = {
    "all": "ALL",
    "chat": "CHAT",
    "coding": "CODE",
    "vision": "VIS",
    "reasoning": "RSN",
    "math": "MATH",
    "embedding": "EMB",
    "general": "GEN",
}

SORT_COMPACT_TAGS = {
    "score": "SCORE",
    "downloads": "DL",
    "name": "NAME",
}

FIT_COMPACT_TAGS = {
    "all": "ALL",
    "fit": "FIT",
    "partial": "PART",
    "nofit": "NO",
}

PROVIDER_COMPACT_TAGS = {
    "Ollama": "OL",
    "Hugging Face": "HF",
    "LM Studio": "LM",
    "Docker": "DK",
    "MLX": "MLX",
}


def option_label(options, key: str, default: str) -> str:
    """Return the display label for *key* in an (key, label) option list."""
    return next((label for option_key, label in options if option_key == key), default)


def provider_compact_tag(label: str) -> str:
    """Return a short stable tag for a provider label in compact mode."""
    return PROVIDER_COMPACT_TAGS.get(label, label[:3].upper() or "-")


def cycle_provider_label(labels, current: str) -> str:
    """Return the next provider label, recovering unknown state to the first option."""
    if not labels:
        return current
    if current not in labels:
        return labels[0]
    return labels[(labels.index(current) + 1) % len(labels)]
