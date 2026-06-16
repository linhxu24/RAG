from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    auto_approve_ingestion: bool = True
    duplicate_ingestion_policy: str = "reject"
    table_classification_threshold: float = 0.85
    enable_llm_router: bool = True
    router_timeout_seconds: int = 0
    router_failure_threshold: int = 2
    router_circuit_breaker_seconds: int = 60
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    database_url: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dental_rag"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    validate_embedding_on_startup: bool = True
    strict_embedding: bool = True
    allow_embedding_fallback: bool = False

    ollama_base_url: str = "http://localhost:11434"
    ollama_generation_model: str = Field(
        default="qwen2.5:7b-instruct",
        validation_alias=AliasChoices("OLLAMA_GENERATION_MODEL", "OLLAMA_LLM_MODEL"),
    )
    ollama_router_model: str = "qwen2.5:3b-instruct"
    ollama_vision_model: str = "llava:latest"
    ollama_timeout_seconds: int = 120
    ollama_keep_alive: str = "30m"
    ollama_num_predict: int = 768

    dense_top_k: int = 20
    dense_min_score: float = 0.25
    sparse_top_k: int = 20
    sparse_trigram_threshold: float = 0.2
    sparse_min_fts_rank: float = 0.001
    sparse_max_per_source: int = 10
    rrf_k: int = 60
    rrf_max_per_source: int = 4
    structured_rrf_weight: float = 1.5
    dense_rrf_weight: float = 1.0
    sparse_rrf_weight: float = 1.0
    rerank_top_n: int = 20
    final_top_k: int = 5
    enable_hyde: bool = False
    hyde_timeout_seconds: int = 15
    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    preload_reranker_on_startup: bool = True
    max_context_chars: int = 16_000
    max_context_items_per_source: int = 4
    confidence_threshold: float = 0.65
    structured_direct_threshold: float = 0.9
    faq_direct_threshold: float = 0.9
    entity_match_threshold: float = 0.42
    entity_ambiguity_margin: float = 0.08
    json_retry_count: int = 1

    asset_storage_dir: Path = Field(
        default=Path("assets"),
        validation_alias=AliasChoices("ASSET_STORAGE_DIR", "ASSETS_DIR"),
    )
    asset_public_base_url: str = Field(
        default="/assets",
        validation_alias=AliasChoices("ASSET_PUBLIC_BASE_URL", "PUBLIC_ASSETS_BASE_URL"),
    )
    upload_dir: Path = Path("uploads")

    enable_langfuse: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    eval_dataset_path: Path = Path("eval_datasets/dental_basic_eval.jsonl")

    @computed_field
    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def ensure_directories(self) -> None:
        self.asset_storage_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
