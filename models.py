from typing import TypedDict


class ModelResult(TypedDict, total=False):
    """A single model entry returned by a provider search.

    All fields are optional (``total=False``) because different providers
    populate different subsets.  Consumers should use ``.get()`` or
    ``model.get("field", default)`` rather than direct key access.
    """

    # -- Core identity --
    inst: str         # Installation indicator markup (e.g. "[green]✔[/green]")
    source: str       # Provider label: "Ollama" or "Hugging Face"
    provider: str     # Human-readable provider name
    id: str           # Unique model identifier / HuggingFace repo id
    name: str         # Short display name
    publisher: str    # Model author or organisation

    # -- Capability metadata --
    params: str       # Parameter count string (e.g. "8B")
    use_case: str     # Rich-markup use-case label (e.g. "[bold blue]Coding[/bold blue]")
    use_case_key: str # Plain use-case key (e.g. "coding", "chat", "vision")
    score: str        # Popularity score markup
    quant: str        # Quantisation level (e.g. "Q4_K_M", "GGUF")
    mode: str         # Inference mode markup (e.g. "[green]GPU[/green]")
    fit: str          # Hardware-fit label markup
    size: str         # Human-readable model size (e.g. "4.8 GB")
    size_source: str  # How size was obtained: "exact" or "estimated"

    # -- Popularity / hidden-gem --
    likes: int        # Like / star count (HuggingFace)
    downloads: int    # Total download count
    is_hidden_gem: bool  # True when classified as a low-profile high-quality model
    gem_score: float  # Hidden-gem ranking score (higher = more gem-like)

    # -- Download state --
    target_file: str      # Target filename for HF file downloads
    download_state: str   # "idle" | "queued" | "downloading" | "completed" | "failed" | "cancelled"
    download_label: str   # Human-readable download state label
    download_detail: str  # Extra detail: progress %, elapsed time, or error message
