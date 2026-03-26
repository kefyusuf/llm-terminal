"""Helpers for provider-aware search and pagination orchestration."""

from __future__ import annotations

from collections.abc import Sequence


def providers_from_filter(current_filter: str) -> list[str]:
    """Return provider slugs from UI filter label."""
    if current_filter == "Hugging Face":
        return ["huggingface"]
    return ["ollama"]


def is_hf_provider_selection(providers: Sequence[str]) -> bool:
    """Return ``True`` when selection targets Hugging Face only."""
    return list(providers) == ["huggingface"]


def provider_display_name(providers: Sequence[str]) -> str:
    """Return user-facing provider name for status messages."""
    return "Hugging Face" if is_hf_provider_selection(providers) else "Ollama"


def build_query_key(providers: Sequence[str], query: str, current_page: int) -> str:
    """Build cache key for search request."""
    normalized_query = query.lower()
    if is_hf_provider_selection(providers):
        return f"hf:{normalized_query}:page{current_page}"
    return f"ollama:{normalized_query}"


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
    if "ollama" in providers:
        return f"Ollama: {result_count} results (pagination not supported by Ollama.com)"
    return f"HF: {result_count} results, has_more={has_more_pages}, page={current_page}"


def page_info_suffix(current_page: int) -> str:
    """Return optional page suffix for completion status."""
    if current_page > 0:
        return f" (Page {current_page + 1})"
    return ""
