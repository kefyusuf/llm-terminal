import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIMODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cache settings
    cache_db_path: Path = Field(default_factory=lambda: Path(__file__).parent / "cache.db")
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
    hf_token: str | None = None

    # Ollama settings
    ollama_search_limit: int = 20
    ollama_api_base: str = "http://localhost:11434"
    ollama_timeout: int = 5

    # Download service settings
    download_service_host: str = "127.0.0.1"
    download_service_port: int = 8765
    download_history_limit: int = 50
    download_history_refresh_interval: float = 0.5

    # Hardware monitoring
    hardware_poll_interval: float = 3.0
    ollama_status_poll_interval: float = 0.5

    # UI settings
    ui_download_poll_interval: float = 1.0


settings = Settings()
