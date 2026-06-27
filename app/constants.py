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
    "memory_load",
    "task_planning",
    "entity_span_extraction",
    "context_binding",
    "entity_resolution",
    "task_canonicalization",
    "bound_task_consistency",
    "router_intent",
    "entity_extraction",
    "retrieval_planning",
    "tool_execution",
    "query_rewrite_hyde",
    "structured_retrieval",
    "dense_retrieval",
    "sparse_retrieval",
    "rrf_fusion",
    "reranker",
    "evidence_merging",
    "evidence_consistency",
    "context_builder",
    "asset_resolver",
    "prompt_builder",
    "synthesis_generation",
    "llm_generation",
    "json_validation",
    "generation_fallback",
    "interest_state_update",
    "suggestion_generation",
    "suggestion_consistency",
    "memory_save",
    "response_rendering",
)
