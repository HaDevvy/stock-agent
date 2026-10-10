from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -------------------------
    # Application
    # -------------------------
    app_name: str = "stock-agent"
    debug: bool = False

    # -------------------------
    # Database
    # -------------------------
    database_url: str

    # -------------------------
    # CODAL Data
    # -------------------------
    codal_data_source_path: str

    # -------------------------
    # 9Router / LLM
    # -------------------------
    nine_router_api_key: str
    nine_router_base_url: str
    model_name: str

    # -------------------------
    # Technical Agent
    # -------------------------
    technical_lookback_days: int = 180

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()