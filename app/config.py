from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ROUTER_TIMEOUT_SECONDS = 30
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120
DEFAULT_OLLAMA_ROUTER_TIMEOUT_SECONDS = 30
DEFAULT_OLLAMA_GENERATION_TIMEOUT_SECONDS = 120
DEFAULT_OPENAI_TIMEOUT_SECONDS = 45
DEFAULT_OPENAI_ROUTER_TIMEOUT_SECONDS = 15
DEFAULT_OPENAI_GENERATION_TIMEOUT_SECONDS = 45


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
    router_timeout_seconds: int = DEFAULT_ROUTER_TIMEOUT_SECONDS
    router_failure_threshold: int = 2
    router_circuit_breaker_seconds: int = 60
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    enable_multi_task_planner: bool = True
    enable_plan_review: bool = True
    enable_evidence_synthesis: bool = True
    enable_query_rewrite: bool = True
    query_rewrite_model: str | None = None
    query_rewrite_history_turns: int = 6
    query_rewrite_timeout_s: float = 8.0
    enable_gliner_ner: bool = True
    gliner_model: str = "urchade/gliner_multi-v2.1"
    gliner_threshold: float = 0.4
    gliner_device: str = "cpu"
    preload_gliner_on_startup: bool = True
    enable_context_binder: bool = True
    context_binder_strict_follow_up: bool = True
    context_binder_trace_decisions: bool = True
    max_sub_queries: int = 3
    max_evidence_items: int = 12
    conversation_history_turns: int = 8
    enable_contextual_suggestions: bool = True
    max_contextual_suggestions: int = 3
    suggestion_history_limit: int = 12

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

    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_generation_model: str = Field(
        default="qwen2.5:14b-instruct-q4_K_M",
        validation_alias=AliasChoices("OLLAMA_GENERATION_MODEL", "OLLAMA_LLM_MODEL"),
    )
    ollama_router_model: str = "qwen2.5:7b-instruct"
    ollama_vision_model: str = "llava:latest"
    ollama_timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    ollama_router_timeout_seconds: int = DEFAULT_OLLAMA_ROUTER_TIMEOUT_SECONDS
    ollama_generation_timeout_seconds: int = DEFAULT_OLLAMA_GENERATION_TIMEOUT_SECONDS
    ollama_keep_alive: str = "30m"
    ollama_num_predict: int = 2048
    ollama_router_num_predict: int = 512
    ollama_generation_num_predict: int = 2048
    ollama_router_num_ctx: int = 8_192
    ollama_generation_num_ctx: int = 16_384

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_router_model: str = "gpt-4.1-nano"
    openai_generation_model: str = "gpt-4.1"
    openai_timeout_seconds: int = DEFAULT_OPENAI_TIMEOUT_SECONDS
    openai_router_timeout_seconds: int = DEFAULT_OPENAI_ROUTER_TIMEOUT_SECONDS
    openai_generation_timeout_seconds: int = DEFAULT_OPENAI_GENERATION_TIMEOUT_SECONDS
    openai_max_tokens: int = 2048
    openai_router_max_tokens: int = 512
    openai_generation_max_tokens: int = 2048

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

    @property
    def llm_router_model(self) -> str:
        if self.llm_provider.lower().strip() == "openai":
            return self.openai_router_model
        return self.ollama_router_model

    @property
    def llm_generation_model(self) -> str:
        if self.llm_provider.lower().strip() == "openai":
            return self.openai_generation_model
        return self.ollama_generation_model

    @property
    def llm_query_rewrite_model(self) -> str:
        configured = (self.query_rewrite_model or "").strip()
        return configured or self.llm_router_model

    @property
    def ollama_request_timeout_seconds(self) -> int:
        return _positive_timeout(
            self.ollama_timeout_seconds,
            default=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        )

    @property
    def openai_request_timeout_seconds(self) -> int:
        return _positive_timeout(
            self.openai_timeout_seconds,
            default=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        )

    @property
    def llm_router_timeout_seconds(self) -> int:
        if self.llm_provider.lower().strip() == "openai":
            return _positive_timeout(
                self.openai_router_timeout_seconds,
                self.openai_timeout_seconds,
                default=DEFAULT_OPENAI_ROUTER_TIMEOUT_SECONDS,
            )
        return _positive_timeout(
            self.ollama_router_timeout_seconds,
            self.ollama_timeout_seconds,
            default=DEFAULT_OLLAMA_ROUTER_TIMEOUT_SECONDS,
        )

    @property
    def llm_generation_timeout_seconds(self) -> int:
        if self.llm_provider.lower().strip() == "openai":
            return _positive_timeout(
                self.openai_generation_timeout_seconds,
                self.openai_timeout_seconds,
                default=DEFAULT_OPENAI_GENERATION_TIMEOUT_SECONDS,
            )
        return _positive_timeout(
            self.ollama_generation_timeout_seconds,
            self.ollama_timeout_seconds,
            default=DEFAULT_OLLAMA_GENERATION_TIMEOUT_SECONDS,
        )

    @property
    def llm_router_num_predict(self) -> int:
        if self.llm_provider.lower().strip() == "openai":
            return self.openai_router_max_tokens
        return self.ollama_router_num_predict

    @property
    def llm_generation_num_predict(self) -> int:
        if self.llm_provider.lower().strip() == "openai":
            return self.openai_generation_max_tokens
        return self.ollama_generation_num_predict

    @property
    def llm_router_num_ctx(self) -> int | None:
        if self.llm_provider.lower().strip() == "openai":
            return None
        return self.ollama_router_num_ctx

    @property
    def llm_generation_num_ctx(self) -> int | None:
        if self.llm_provider.lower().strip() == "openai":
            return None
        return self.ollama_generation_num_ctx


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _positive_timeout(*values: int | None, default: int) -> int:
    for value in values:
        if value is not None and int(value) > 0:
            return int(value)
    return default
