import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.constants import Intent
from app.evaluation.eval_router import evaluate_router
from app.generation.ollama_client import OllamaResponse
from app.retrieval.router import IntentRouter


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Xin chào", Intent.GREETING),
        ("Địa chỉ phòng khám ở đâu?", Intent.CLINIC_INFO),
        ("Cho tôi bảng sản phẩm", Intent.PRODUCT_LIST),
        ("Xem dịch vụ tẩy trắng răng", Intent.SERVICE_DETAIL),
        ("So sánh sản phẩm Oral-B với Sensodyne", Intent.PRODUCT_COMPARE),
        ("Tại sao răng bị ê buốt?", Intent.FAQ),
        ("asdfghjkl", Intent.UNKNOWN),
    ],
)
def test_rule_router(query, expected):
    assert IntentRouter().route(query).intent == expected


def test_known_product_name_routes_to_detail():
    result = IntentRouter().route(
        "Cho tôi thông tin Pro 500",
        known_products=["Oral-B Pro 500"],
    )
    assert result.intent == Intent.PRODUCT_DETAIL
    assert result.entities == ["Oral-B Pro 500"]


def test_common_greeting_variant_routes_to_greeting():
    assert IntentRouter().route("Chào buổi sáng phòng khám").intent == Intent.GREETING


def test_post_treatment_question_prefers_faq():
    result = IntentRouter().route(
        "Sau khi nhổ răng tôi cần kiêng gì?",
        known_services=["Nhổ răng"],
    )
    assert result.intent == Intent.FAQ


def test_post_treatment_paraphrase_prefers_faq():
    result = IntentRouter().route(
        "Sau nhổ răng tôi nên ăn uống và vệ sinh như thế nào?",
        known_services=["Nhổ răng"],
    )

    assert result.intent == Intent.FAQ


def test_known_service_matching_does_not_add_unrelated_fuzzy_entity():
    result = IntentRouter().route(
        "Dịch vụ trồng răng implant mất bao lâu?",
        known_services=["Tẩy trắng răng", "Trồng răng implant"],
    )
    assert result.intent == Intent.SERVICE_DETAIL
    assert result.entities == ["Trồng răng implant"]


def test_known_entity_mentions_are_deduplicated():
    result = IntentRouter().route(
        "Chi phí nhổ răng khôn là bao nhiêu?",
        known_services=["Nhổ răng khôn", "Nhổ răng khôn"],
    )

    assert result.entities == ["Nhổ răng khôn"]


def test_product_list_wording_routes_to_list():
    result = IntentRouter().route("Phòng khám đang bán những sản phẩm nào?")
    assert result.intent == Intent.PRODUCT_LIST


def test_service_list_filter_wording_routes_to_list():
    result = IntentRouter().route(
        "Dịch vụ nào liên quan đến implant?",
        known_services=["Cấy ghép Implant đơn lẻ", "Tẩy trắng răng tại phòng khám"],
    )

    assert result.intent == Intent.SERVICE_LIST


def test_product_filter_wording_routes_to_list():
    result = IntentRouter().route(
        "Sản phẩm nào dưới 2 triệu?",
        known_products=["Bàn chải điện Oral-B Pro 500"],
    )

    assert result.intent == Intent.PRODUCT_LIST


def test_router_evaluation_uses_known_business_entities():
    metrics = evaluate_router(
        [
            {
                "query": "Cho tôi thông tin AquaJet Mini Water Flosser",
                "expected_intent": "PRODUCT_DETAIL",
            }
        ],
        known_products=["AquaJet Mini Water Flosser"],
    )

    assert metrics["accuracy"] == 1.0


def test_optional_llm_router_uses_validated_json_output():
    ollama = AsyncMock()
    settings = Settings(enable_llm_router=True)
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "intent": "FAQ",
                "confidence": 0.92,
                "entities": [],
                "needs_rag": True,
                "needs_clarification": False,
            }
        ),
        latency_ms=12,
        model="router",
    )

    result = asyncio.run(
        IntentRouter().route_with_optional_llm(
            "Tư vấn thêm về sức khỏe răng miệng",
            settings,
            ollama,
        )
    )

    assert result.intent == Intent.FAQ
    assert result.source == "ollama"
    assert result.llm_attempted is True
    assert result.llm_prompt_chars is not None
    assert result.llm_metadata["model"] == "router"
    ollama.generate.assert_awaited_once()
    assert ollama.generate.await_args.kwargs["timeout_seconds"] == (
        settings.llm_router_timeout_seconds
    )
    assert ollama.generate.await_args.kwargs["timeout_seconds"] > 0


def test_optional_llm_router_falls_back_and_opens_circuit():
    ollama = AsyncMock()
    ollama.generate.side_effect = TimeoutError("router timeout")
    settings = Settings(
        enable_llm_router=True,
        router_failure_threshold=2,
        router_circuit_breaker_seconds=60,
    )
    router = IntentRouter()

    first = asyncio.run(
        router.route_with_optional_llm("Câu hỏi không rõ một", settings, ollama)
    )
    second = asyncio.run(
        router.route_with_optional_llm("Câu hỏi không rõ hai", settings, ollama)
    )
    third = asyncio.run(
        router.route_with_optional_llm("Câu hỏi không rõ ba", settings, ollama)
    )

    assert first.source == "rules_fallback"
    assert second.source == "rules_fallback"
    assert third.source == "rules_circuit_open"
    assert third.llm_attempted is False
    assert ollama.generate.await_count == 2


def test_llm_router_failure_uses_safe_fallback_not_full_rule_router():
    ollama = AsyncMock()
    ollama.generate.side_effect = TimeoutError("router timeout")

    result = asyncio.run(
        IntentRouter().route_with_optional_llm(
            "Dịch vụ tẩy trắng răng giá bao nhiêu?",
            Settings(enable_llm_router=True),
            ollama,
            known_services=["Tẩy trắng răng tại phòng khám"],
        )
    )

    assert result.source == "rules_fallback"
    assert result.intent == Intent.UNKNOWN
    assert result.needs_clarification is True
    assert result.llm_metadata["fallback_kind"] == "safe_fallback"


def test_llm_router_retries_once_to_fix_invalid_json():
    ollama = AsyncMock()
    ollama.generate.side_effect = [
        OllamaResponse(
            text='{"intent":"FAQ","confidence":1.2,"entities":[]}',
            latency_ms=10,
            model="router",
        ),
        OllamaResponse(
            text=json.dumps(
                {
                    "intent": "FAQ",
                    "confidence": 0.91,
                    "entities": [],
                    "needs_rag": True,
                    "needs_clarification": False,
                }
            ),
            latency_ms=11,
            model="router",
        ),
    ]

    result = asyncio.run(
        IntentRouter().route_with_optional_llm(
            "Hôi miệng có phải do sâu răng không?",
            Settings(enable_llm_router=True),
            ollama,
        )
    )

    assert result.intent == Intent.FAQ
    assert result.source == "ollama"
    assert result.llm_metadata["validation"]["retried"] is True
    assert ollama.generate.await_count == 2


def test_high_confidence_rule_still_uses_llm_router_when_enabled():
    ollama = AsyncMock()
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "intent": "PRODUCT_LIST",
                "confidence": 0.97,
                "entities": [],
                "question_type": "structured_lookup",
                "answer_strategy": "direct_sql",
                "needs_rag": False,
                "needs_clarification": False,
            }
        ),
        latency_ms=10,
        model="router",
    )

    result = asyncio.run(
        IntentRouter().route_with_optional_llm(
            "Cho tôi danh sách sản phẩm",
            Settings(enable_llm_router=True),
            ollama,
        )
    )

    assert result.intent == Intent.PRODUCT_LIST
    assert result.source == "ollama"
    assert result.llm_attempted is True
    ollama.generate.assert_awaited_once()


def test_product_category_sort_query_routes_to_product_list():
    result = IntentRouter().route(
        "Bàn chải điện giá từ thấp đến cao",
        known_product_categories=["Bàn chải điện"],
    )

    assert result.intent == Intent.PRODUCT_LIST


def test_llm_router_normalizes_structured_entity_objects():
    ollama = AsyncMock()
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "intent": "FAQ",
                "confidence": 0.92,
                "entities": [
                    {"type": "service", "name": "Cạo vôi răng", "role": "topic"}
                ],
                "question_type": "knowledge_or_advice",
                "answer_strategy": "faq_retrieval",
                "needs_rag": True,
                "needs_clarification": False,
                "reason_code": "service_entity_with_risk_question",
            }
        ),
        latency_ms=33,
        model="router",
    )

    result = asyncio.run(
        IntentRouter().route_with_optional_llm(
            "Cạo vôi răng có làm yếu răng không",
            Settings(enable_llm_router=True),
            ollama,
        )
    )

    assert result.intent == Intent.FAQ
    assert result.entities == ["Cạo vôi răng"]
    assert result.entity_details == [
        {"type": "service", "name": "Cạo vôi răng", "role": "topic"}
    ]
    assert result.question_type == "knowledge_or_advice"
    assert result.answer_strategy == "faq_retrieval"
    assert result.reason_code == "service_entity_with_risk_question"


def test_llm_router_accepts_unknown_entity_types_and_null_names():
    ollama = AsyncMock()
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "intent": "FAQ",
                "confidence": 0.95,
                "entities": [
                    {
                        "type": "symptom_or_condition",
                        "name": "Hôi miệng",
                        "role": "topic",
                    },
                    {"type": "product", "name": None, "role": "scope"},
                ],
                "question_type": "knowledge_or_advice",
                "answer_strategy": "faq_retrieval",
                "needs_rag": True,
                "needs_clarification": False,
            }
        ),
        latency_ms=21,
        model="router",
    )

    result = asyncio.run(
        IntentRouter().route_with_optional_llm(
            "Hôi miệng có phải do sâu răng không?",
            Settings(enable_llm_router=True),
            ollama,
        )
    )

    assert result.intent == Intent.FAQ
    assert result.entities == ["Hôi miệng"]
    assert result.entity_details == [
        {"type": "unknown", "name": "Hôi miệng", "role": "topic"},
        {"type": "product", "name": None, "role": "scope"},
    ]


def test_rule_fallback_handles_no_accent_faq_conflict():
    result = IntentRouter().route(
        "dich vu nho rang khon co dau khong",
        known_services=["Nhổ răng khôn"],
    )

    assert result.intent == Intent.FAQ


def test_rule_fallback_handles_no_accent_service_detail():
    result = IntentRouter().route(
        "dich vu nho rang khon gia bao nhieu",
        known_services=["Nhổ răng khôn"],
    )

    assert result.intent == Intent.SERVICE_DETAIL
    assert result.entities == ["Nhổ răng khôn"]
