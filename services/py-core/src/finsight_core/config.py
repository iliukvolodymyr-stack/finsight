from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_ENV: Literal["local", "staging", "prod", "demo"] = "local"
    DEMO_MODE: bool = False
    LOG_LEVEL: str = "INFO"
    API_BASE_URL: str = "http://localhost:8000"
    CORS_ORIGINS: str = "http://localhost:5173,http://tauri.localhost,tauri://localhost"

    # Auth
    AUTH_MODE: Literal["single_user", "jwt"] = "single_user"
    JWT_SECRET: str = "change-me-32-chars-min-secure-secret-key-12345"
    ACCESS_TOKEN_TTL_MIN: int = 15
    REFRESH_TOKEN_TTL_DAYS: int = 30

    # Data & Storage
    DATABASE_URL: str = "postgresql+asyncpg://finsight:finsight@localhost:5432/finsight"
    REDIS_URL: str = "redis://localhost:6379/0"
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    ARTIFACTS_DIR: str = "./var/artifacts"
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minio"
    S3_SECRET_KEY: str = "minio123"
    S3_BUCKET_MODELS: str = "models"
    S3_BUCKET_DATASETS: str = "datasets"
    S3_BUCKET_REPORTS: str = "reports"

    # Market data
    BINANCE_REST: str = "https://api.binance.com"
    BINANCE_WS: str = "wss://stream.binance.com:9443"
    ALPHAVANTAGE_API_KEY: str = ""
    YAHOO_ENABLED: bool = True

    # News & LLM
    CRYPTOPANIC_API_KEY: str = ""
    TELEGRAM_API_ID: str = ""
    TELEGRAM_API_HASH: str = ""
    OPENROUTER_API_KEY: str = ""
    LLM_SENTIMENT_MODEL: str = "deepseek/deepseek-v4.1-flash"
    LLM_ASSISTANT_MODEL: str = "google/gemini-3.8-flash"
    LLM_DAILY_BUDGET_USD: float = 2.00
    LLM_CACHE_TTL_DAYS: int = 30

    # Mobile
    EXPO_ACCESS_TOKEN: str = ""

    # Observability
    SENTRY_DSN: str = ""
    METRICS_TOKEN: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @model_validator(mode="after")
    def validate_auth_mode_env(self) -> "Settings":
        if self.AUTH_MODE == "single_user" and self.APP_ENV not in ("local", "demo"):
            raise ValueError(
                "AUTH_MODE=single_user is only permitted when APP_ENV in ('local', 'demo')"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
