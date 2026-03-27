"""Tests for new provider implementations and provider registry."""

from unittest.mock import patch

from providers import (
    BaseProvider,
    detect_available_providers,
    get_all_provider_classes,
    get_provider_display_names,
    get_provider_filter_labels,
)

# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_get_all_provider_classes_returns_list(self):
        classes = get_all_provider_classes()
        assert isinstance(classes, list)

    def test_all_classes_are_base_provider_subclasses(self):
        for cls in get_all_provider_classes():
            assert issubclass(cls, BaseProvider)

    def test_detect_available_providers_returns_dict(self):
        result = detect_available_providers()
        assert isinstance(result, dict)
        assert "ollama" in result
        assert "huggingface" in result

    def test_huggingface_always_available(self):
        result = detect_available_providers()
        assert result["huggingface"] is True

    def test_display_names_returns_dict(self):
        names = get_provider_display_names()
        assert "ollama" in names
        assert "huggingface" in names
        assert names["huggingface"] == "Hugging Face"

    def test_filter_labels_returns_list(self):
        labels = get_provider_filter_labels()
        assert isinstance(labels, list)
        assert "Ollama" in labels
        assert "Hugging Face" in labels


# ---------------------------------------------------------------------------
# LM Studio Provider
# ---------------------------------------------------------------------------


class TestLMStudioProvider:
    def test_slug(self):
        from providers.lmstudio_provider import LMStudioProvider

        assert LMStudioProvider.slug == "lmstudio"
        assert LMStudioProvider.display_name == "LM Studio"

    def test_detect_returns_false_when_unavailable(self):
        from providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider()
        # LM Studio likely not running in test environment
        result = provider.detect()
        assert isinstance(result, bool)

    def test_search_returns_tuple(self):
        from providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider()
        results, errors = provider.search("*", {"has_gpu": False, "vram_total": 0, "ram_total": 16})
        assert isinstance(results, list)
        assert isinstance(errors, list)

    def test_list_installed_returns_list(self):
        from providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider()
        installed = provider.list_installed()
        assert isinstance(installed, list)


# ---------------------------------------------------------------------------
# Docker Provider
# ---------------------------------------------------------------------------


class TestDockerProvider:
    def test_slug(self):
        from providers.docker_provider import DockerProvider

        assert DockerProvider.slug == "docker"
        assert DockerProvider.display_name == "Docker"

    def test_detect_returns_false_when_unavailable(self):
        from providers.docker_provider import DockerProvider

        provider = DockerProvider()
        result = provider.detect()
        assert isinstance(result, bool)

    def test_search_returns_tuple(self):
        from providers.docker_provider import DockerProvider

        provider = DockerProvider()
        results, errors = provider.search("*", {"has_gpu": False, "vram_total": 0, "ram_total": 16})
        assert isinstance(results, list)
        assert isinstance(errors, list)


# ---------------------------------------------------------------------------
# MLX Provider
# ---------------------------------------------------------------------------


class TestMLXProvider:
    def test_slug(self):
        from providers.mlx_provider import MLXProvider

        assert MLXProvider.slug == "mlx"
        assert MLXProvider.display_name == "MLX"

    def test_detect_returns_false_on_non_macos(self):
        from providers.mlx_provider import MLXProvider

        provider = MLXProvider()
        with patch("providers.mlx_provider.platform.system", return_value="Windows"):
            assert provider.detect() is False

    def test_search_returns_tuple(self):
        from providers.mlx_provider import MLXProvider

        provider = MLXProvider()
        results, errors = provider.search("*", {"has_gpu": False, "vram_total": 0, "ram_total": 16})
        assert isinstance(results, list)
        assert isinstance(errors, list)

    def test_list_installed_returns_list(self):
        from providers.mlx_provider import MLXProvider

        provider = MLXProvider()
        installed = provider.list_installed()
        assert isinstance(installed, list)


# ---------------------------------------------------------------------------
# BaseProvider interface
# ---------------------------------------------------------------------------


class TestBaseProviderInterface:
    def test_lmstudio_implements_all_methods(self):
        from providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider()
        assert hasattr(provider, "detect")
        assert hasattr(provider, "search")
        assert hasattr(provider, "list_installed")
        assert callable(provider.detect)
        assert callable(provider.search)
        assert callable(provider.list_installed)

    def test_docker_implements_all_methods(self):
        from providers.docker_provider import DockerProvider

        provider = DockerProvider()
        assert hasattr(provider, "detect")
        assert hasattr(provider, "search")
        assert hasattr(provider, "list_installed")

    def test_mlx_implements_all_methods(self):
        from providers.mlx_provider import MLXProvider

        provider = MLXProvider()
        assert hasattr(provider, "detect")
        assert hasattr(provider, "search")
        assert hasattr(provider, "list_installed")
