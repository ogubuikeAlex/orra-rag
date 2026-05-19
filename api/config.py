from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ORRA RAG API"
    environment: str = "development"
    api_prefix: str = "/v1"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    docs_dir: Path = BASE_DIR / "docs"
    db_dir: Path = BASE_DIR / "db"
    table_name: str = "knowledge"
    auto_index_on_startup: bool = True

    chunk_size_tokens: int = 1200
    chunk_overlap_tokens: int = 150
    retrieval_limit: int = 5
    retrieval_query_type: str = "hybrid"
    reranker_weight: float = 0.7

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_generation_model: str = "gemini-2.0-flash"

    gist_raw_url: str | None = Field(default=None, alias="APP_GIST_RAW_URL")
    gist_records_path: str | None = Field(default=None, alias="APP_GIST_RECORDS_PATH")
    gist_cache_ttl_seconds: int = 300
    request_timeout_seconds: float = 10.0

    allowed_origins: list[str] = Field(
        default_factory=lambda: ["https://orra.xyz/"],
        alias="ALLOWED_ORIGINS",
    )

    @field_validator("docs_dir", "db_dir", mode="before")
    @classmethod
    def _normalize_path(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def gist_enabled(self) -> bool:
        return bool(self.gist_raw_url)

    def ensure_directories(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
