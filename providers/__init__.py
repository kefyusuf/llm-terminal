"""Providers package — pluggable search backends for LLM model discovery.

Provides a unified provider architecture with:
- ``BaseProvider`` — abstract base class for all providers
- ``PROVIDERS`` — registry of all available providers
- Provider detection and dynamic availability checking
"""

from __future__ import annotations

from contextlib import suppress
from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """Abstract base class for model search providers.

    Each provider implements search, detection, and installed-model listing
    for a specific LLM runtime (Ollama, Hugging Face, LM Studio, etc.).
    """

    # Class-level metadata (override in subclasses)
    slug: str = ""  # Internal identifier (e.g., "ollama", "huggingface")
    display_name: str = ""  # Human-readable name (e.g., "Ollama")
    default_host: str = ""  # Default API host

    @abstractmethod
    def detect(self) -> bool:
        """Return True if this provider is available on the system."""

    @abstractmethod
    def search(
        self,
        query: str,
        specs: dict[str, Any],
        limit: int = 15,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Search for models matching *query*.

        Returns:
            ``(results, errors)`` — list of model result dicts and error messages.
        """

    @abstractmethod
    def list_installed(self) -> list[str]:
        """Return list of locally installed model identifiers."""

    def search_with_installed(
        self,
        query: str,
        specs: dict[str, Any],
        limit: int = 15,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Search and mark installed models. Override for custom behavior."""
        return self.search(query, specs, limit=limit, **kwargs)


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------


# Lazy imports to avoid circular dependencies
def _get_ollama_provider():
    from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

    return search_ollama_models, get_installed_ollama_models


def _get_hf_provider():
    from providers.hf_provider import enrich_hf_model_details, search_hf_models

    return search_hf_models, enrich_hf_model_details


def _get_lmstudio_provider():
    from providers.lmstudio_provider import LMStudioProvider

    return LMStudioProvider


def _get_docker_provider():
    from providers.docker_provider import DockerProvider

    return DockerProvider


def _get_mlx_provider():
    from providers.mlx_provider import MLXProvider

    return MLXProvider


def get_all_provider_classes() -> list[type[BaseProvider]]:
    """Return all available provider classes (may fail on non-matching platforms)."""
    providers = []
    for getter in [_get_lmstudio_provider, _get_docker_provider, _get_mlx_provider]:
        with suppress(ImportError, Exception):
            providers.append(getter())
    return providers


def detect_available_providers() -> dict[str, bool]:
    """Detect which providers are available on this system.

    Returns a dict mapping provider slug to availability bool.
    """
    from hardware import check_ollama_running

    available = {
        "ollama": check_ollama_running(),
        "huggingface": True,  # Always available via API
    }

    for provider_cls in get_all_provider_classes():
        try:
            instance = provider_cls()
            available[instance.slug] = instance.detect()
        except Exception:
            pass

    return available


def get_provider_display_names() -> dict[str, str]:
    """Return mapping of provider slug to display name."""
    names = {
        "ollama": "Ollama",
        "huggingface": "Hugging Face",
    }

    for provider_cls in get_all_provider_classes():
        with suppress(Exception):
            names[provider_cls.slug] = provider_cls.display_name

    return names


def get_provider_filter_labels() -> list[str]:
    """Return list of filter labels for the UI provider selector."""
    labels = ["Ollama", "Hugging Face"]

    for provider_cls in get_all_provider_classes():
        try:
            instance = provider_cls()
            if instance.detect():
                labels.append(instance.display_name)
        except Exception:
            pass

    return labels
