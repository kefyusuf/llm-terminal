import shutil
from contextlib import suppress
from pathlib import Path
from typing import Literal

from platformdirs import user_data_path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "ai-model-explorer"


def _legacy_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _default_data_dir() -> Path:
    return Path(user_data_path(APP_NAME, appauthor=False))


def _default_data_file(filename: str) -> Path:
    target = _default_data_dir() / filename
    if target.exists():
        return target

    legacy = _legacy_data_dir() / filename
    if not legacy.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with suppress(FileExistsError):
        shutil.copy2(legacy, target)
    return target


def _default_cache_db_path() -> Path:
    return _default_data_file("cache.db")


def _default_download_db_path() -> Path:
    return _default_data_file("downloads.db")


def _default_hf_models_dir() -> Path:
    """Return the persistent per-user directory used for Hugging Face model files."""
    return _default_data_dir() / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIMODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cache settings
    cache_db_path: Path = Field(default_factory=_default_cache_db_path)
    cache_ttl_seconds: int = 86400
    cache_max_per_source: int = 500

    # Search cache settings
    search_cache_ttl_seconds: int = 90
    search_cache_max_entries: int = 20
    search_cache_ram_threshold_gb: float = 1.0
    search_cache_vram_threshold_gb: float = 1.0

    # HuggingFace settings
    hf_search_limit: int = 15
    hf_search_max_pages: int = 10
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AIMODEL_HF_TOKEN", "HF_TOKEN"),
    )
    hf_models_dir: Path = Field(default_factory=_default_hf_models_dir)

    # Ollama settings
    ollama_search_limit: int = 20
    ollama_api_base: str = "http://localhost:11434"
    ollama_timeout: int = 5

    # Download settings
    download_db_path: Path = Field(default_factory=_default_download_db_path)
    download_history_limit: int = 50
    download_history_refresh_interval: float = 0.9
    download_max_workers: int = 2

    # Hardware monitoring
    hardware_poll_interval: float = 3.0
    ollama_status_poll_interval: float = 1.5

    # UI settings
    ui_download_poll_interval: float = 1.5
    ui_mode: Literal["comfortable", "compact"] = "compact"
    theme: str = "default"


settings = Settings()
