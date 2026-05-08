"""Helpers for provider-aware search and pagination orchestration."""

from __future__ import annotations

from collections.abc import Sequence

# All supported provider slugs
ALL_PROVIDER_SLUGS = ["ollama", "huggingface", "lmstudio", "docker", "mlx"]

# Mapping from UI filter labels to provider slugs
_FILTER_TO_SLUGS: dict[str, list[str]] = {
    "Ollama": ["ollama"],
    "Hugging Face": ["huggingface"],
    "LM Studio": ["lmstudio"],
    "Docker": ["docker"],
    "MLX": ["mlx"],
    "All": ALL_PROVIDER_SLUGS,
}


def providers_from_filter(current_filter: str) -> list[str]:
    """Return provider slugs from UI filter label."""
    return _FILTER_TO_SLUGS.get(current_filter, ["ollama"])


def is_hf_provider_selection(providers: Sequence[str]) -> bool:
    """Return ``True`` when selection targets Hugging Face only."""
    return list(providers) == ["huggingface"]


def is_multi_provider(providers: Sequence[str]) -> bool:
    """Return ``True`` when selection includes multiple providers."""
    return len(providers) > 1


def provider_display_name(providers: Sequence[str]) -> str:
    """Return user-facing provider name for status messages."""
    if len(providers) == 1:
        return {
            "ollama": "Ollama",
            "huggingface": "Hugging Face",
            "lmstudio": "LM Studio",
            "docker": "Docker",
            "mlx": "MLX",
        }.get(providers[0], providers[0].title())
    return "All Providers"


def build_query_key(providers: Sequence[str], query: str, current_page: int) -> str:
    """Build cache key for search request."""
    normalized_query = query.lower()
    provider_key = "+".join(sorted(providers))
    if is_hf_provider_selection(providers):
        return f"hf:{normalized_query}:page{current_page}"
    if len(providers) == 1:
        return f"{providers[0]}:{normalized_query}"
    return f"{provider_key}:{normalized_query}:page{current_page}"


def cache_hit_suffix(providers: Sequence[str], current_page: int) -> str:
    """Return status suffix for cached search hits."""
    if is_hf_provider_selection(providers):
        return f" (cached page {current_page + 1})"
    return " (cached)"


def validate_page_request(page_num: int, max_pages: int) -> tuple[bool, str | None]:
    """Validate requested page index against bounds."""
    if page_num < 0:
        return False, None
    if page_num >= max_pages:
        return False, f"Reached page limit ({max_pages})."
    return True, None


def has_more_pages_for_results(
    providers: Sequence[str],
    hf_result_count: int,
    ollama_result_count: int,
    page_size: int,
) -> bool:
    """Compute whether next-page navigation should be enabled."""
    if "huggingface" in providers and hf_result_count > 0:
        return hf_result_count == page_size
    if "ollama" in providers and ollama_result_count > 0:
        return False
    return False


def provider_result_count(
    providers: Sequence[str], hf_result_count: int, ollama_result_count: int
) -> int:
    """Return result count for currently selected provider."""
    if "huggingface" in providers:
        return hf_result_count
    return ollama_result_count


def provider_search_status(
    providers: Sequence[str],
    result_count: int,
    has_more_pages: bool,
    current_page: int,
) -> str:
    """Build post-search status message for provider context."""
    if len(providers) > 1:
        return f"All: {result_count} results"
    if "ollama" in providers:
        return f"Ollama: {result_count} results (pagination not supported by Ollama.com)"
    if "lmstudio" in providers:
        return f"LM Studio: {result_count} results"
    if "docker" in providers:
        return f"Docker: {result_count} results"
    if "mlx" in providers:
        return f"MLX: {result_count} results"
    return f"HF: {result_count} results, has_more={has_more_pages}, page={current_page}"


def page_info_suffix(current_page: int) -> str:
    """Return optional page suffix for completion status."""
    if current_page > 0:
        return f" (Page {current_page + 1})"
    return ""
