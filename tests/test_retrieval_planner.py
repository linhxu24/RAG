from app.config import Settings
from app.constants import Intent, RetrievalMode
from app.retrieval.entity_resolver import EntityCandidate, EntityResolution
from app.retrieval.planner import RetrievalPlanner
from app.retrieval.router import RouterResult
from app.retrieval.types import RetrievalResult


def _structured(score: float = 1.0) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            source_type="product",
            source_id="product-1",
            text="Sản phẩm mẫu",
            score=score,
        )
    ]


def _resolution(
    *,
    status: str = "resolved",
    score: float = 1.0,
) -> EntityResolution:
    return EntityResolution(
        status=status,
        products=[
            EntityCandidate(
                entity_type="product",
                entity_id="product-1",
                name="Oral-B",
                score=score,
                match_type="contained",
            )
        ],
    )


def test_low_confidence_route_requires_clarification():
    plan = RetrievalPlanner(Settings()).plan(
        query="Tôi cần tư vấn",
        routed=RouterResult(
            Intent.UNKNOWN,
            0.4,
            needs_clarification=True,
        ),
        entities=EntityResolution(),
        structured=[],
    )

    assert plan.mode == RetrievalMode.CLARIFY
    assert plan.run_dense is False
    assert plan.run_sparse is False


def test_chitchat_uses_no_rag_llm_mode():
    plan = RetrievalPlanner(Settings()).plan(
        query="Bạn là ai?",
        routed=RouterResult(Intent.CHITCHAT, 0.9, source="ollama"),
        entities=EntityResolution(),
        structured=[],
    )

    assert plan.mode == RetrievalMode.NO_RAG_LLM
    assert plan.run_structured is False
    assert plan.uses_hybrid is False


def test_high_confidence_unknown_dental_query_uses_hybrid():
    plan = RetrievalPlanner(Settings()).plan(
        query="Tư vấn tình trạng nướu bị sưng",
        routed=RouterResult(Intent.UNKNOWN, 0.8),
        entities=EntityResolution(),
        structured=[],
    )

    assert plan.mode == RetrievalMode.HYBRID
    assert plan.uses_hybrid is True


def test_fuzzy_product_match_requires_clarification_without_rag():
    plan = RetrievalPlanner(Settings()).plan(
        query="Thông tin oral b",
        routed=RouterResult(Intent.PRODUCT_DETAIL, 0.9),
        entities=_resolution(status="partial", score=0.78),
        structured=_structured(0.78),
    )

    assert plan.mode == RetrievalMode.CLARIFY
    assert plan.run_structured is True
    assert plan.uses_hybrid is False


def test_high_confidence_product_match_skips_hybrid():
    plan = RetrievalPlanner(Settings()).plan(
        query="Thông tin Oral-B",
        routed=RouterResult(Intent.PRODUCT_DETAIL, 0.95),
        entities=_resolution(),
        structured=_structured(),
    )

    assert plan.mode == RetrievalMode.STRUCTURED_ONLY
    assert plan.uses_hybrid is False


def test_unique_resolved_entity_with_authoritative_sql_match_skips_hybrid():
    plan = RetrievalPlanner(Settings()).plan(
        query="Cho tôi xem bàn chải đánh răng",
        routed=RouterResult(Intent.PRODUCT_DETAIL, 0.91),
        entities=_resolution(status="resolved", score=0.52),
        structured=_structured(1.0),
    )

    assert plan.mode == RetrievalMode.STRUCTURED_ONLY
    assert plan.reason == "resolved_entity_matches_structured_record"
    assert plan.uses_hybrid is False


def test_resolved_entity_does_not_trust_mismatched_structured_record():
    structured = _structured(1.0)
    structured[0].source_id = "different-product"
    plan = RetrievalPlanner(Settings()).plan(
        query="Cho tôi xem bàn chải đánh răng",
        routed=RouterResult(Intent.PRODUCT_DETAIL, 0.91),
        entities=_resolution(status="resolved", score=0.52),
        structured=structured,
    )

    assert plan.mode == RetrievalMode.CLARIFY
    assert plan.uses_hybrid is False


def test_faq_exact_match_can_use_structured_only():
    plan = RetrievalPlanner(Settings()).plan(
        query="Tại sao răng ê buốt?",
        routed=RouterResult(Intent.FAQ, 0.9),
        entities=EntityResolution(),
        structured=[
            RetrievalResult(
                source_type="faq",
                source_id="faq-1",
                text="FAQ",
                score=0.95,
            )
        ],
    )

    assert plan.mode == RetrievalMode.STRUCTURED_ONLY


def test_faq_fallback_searches_without_hyde_or_reranker():
    plan = RetrievalPlanner(
        Settings(enable_hyde=True, enable_reranker=True)
    ).plan(
        query="Tôi nên chăm sóc sau nhổ răng thế nào?",
        routed=RouterResult(Intent.FAQ, 0.9),
        entities=EntityResolution(),
        structured=[],
    )

    assert plan.uses_hybrid is True
    assert plan.run_hyde is False
    assert plan.use_reranker is False
