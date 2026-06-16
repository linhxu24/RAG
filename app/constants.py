from enum import StrEnum


class Intent(StrEnum):
    GREETING = "GREETING"
    CHITCHAT = "CHITCHAT"
    CLINIC_INFO = "CLINIC_INFO"
    FAQ = "FAQ"
    PRODUCT_LIST = "PRODUCT_LIST"
    PRODUCT_DETAIL = "PRODUCT_DETAIL"
    PRODUCT_COMPARE = "PRODUCT_COMPARE"
    SERVICE_LIST = "SERVICE_LIST"
    SERVICE_DETAIL = "SERVICE_DETAIL"
    UNKNOWN = "UNKNOWN"


class RetrievalMode(StrEnum):
    TEMPLATE = "TEMPLATE"
    NO_RAG_LLM = "NO_RAG_LLM"
    DIRECT_SQL = "DIRECT_SQL"
    STRUCTURED_ONLY = "STRUCTURED_ONLY"
    STRUCTURED_THEN_HYBRID = "STRUCTURED_THEN_HYBRID"
    HYBRID = "HYBRID"
    CLARIFY = "CLARIFY"


ACTIVE_STATUS = "active"
REVIEW_STATUS = "review_required"

TRACE_STEPS = (
    "router_intent",
    "entity_extraction",
    "retrieval_planning",
    "query_rewrite_hyde",
    "structured_retrieval",
    "dense_retrieval",
    "sparse_retrieval",
    "rrf_fusion",
    "reranker",
    "context_builder",
    "asset_resolver",
    "prompt_builder",
    "llm_generation",
    "json_validation",
    "response_rendering",
)
