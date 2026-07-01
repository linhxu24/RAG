import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from app.config import Settings
from app.constants import Intent
from app.generation.generator import GroundedGenerator
from app.generation.ollama_client import OllamaResponse


def test_product_fallback_uses_only_authoritative_product_context():
    generator = GroundedGenerator(Settings(), Mock())
    context = {
        "items": [
            {
                "source_type": "product",
                "source_id": "product-1",
                "text": "Sản phẩm đúng [asset:correct]",
                "raw_json": {"name": "Sản phẩm đúng", "asset_id": "asset-1"},
                "source": {},
            },
            {
                "source_type": "chunk",
                "source_id": "chunk-1",
                "text": "Sản phẩm khác [asset:wrong]",
                "raw_json": {},
                "source": {},
            },
        ],
        "total_chars": 100,
    }

    response = generator.fallback_from_context(
        intent=Intent.PRODUCT_DETAIL,
        confidence=0.8,
        context=context,
    )

    assert "Sản phẩm đúng" in response.result.text
    assert "Sản phẩm khác" not in response.result.text
    assert len(response.result.items) == 1
    assert response.result.items[0].id == "product-1"
    assert response.answer_type == "fallback"
    assert response.degraded is True


def test_faq_fallback_uses_top_faq_instead_of_related_chunks():
    generator = GroundedGenerator(Settings(), Mock())
    context = {
        "items": [
            {
                "source_type": "faq",
                "source_id": "faq-1",
                "text": "Câu hỏi\nTrả lời đúng",
                "raw_json": {"question": "Câu hỏi", "answer": "Trả lời đúng"},
                "source": {},
            },
            {
                "source_type": "chunk",
                "source_id": "chunk-1",
                "text": "Nội dung liên quan nhưng không phải câu trả lời",
                "raw_json": {},
                "source": {},
            },
        ],
        "total_chars": 100,
    }

    response = generator.fallback_from_context(
        intent=Intent.FAQ,
        confidence=0.8,
        context=context,
    )

    assert "Trả lời đúng" in response.result.text
    assert len(response.result.items) == 1
    assert response.result.items[0].id == "faq-1"
    assert response.entities[0].name == "Câu hỏi"
    assert response.answer_type == "fallback"
    assert response.degraded is True


def test_direct_response_formats_product_source_by_capability_shape():
    generator = GroundedGenerator(Settings(), Mock())
    context = {
        "items": [
            {
                "source_type": "product",
                "source_id": "product-1",
                "text": "Máy tăm nước AquaJet. Giá: 850000",
                "raw_json": {
                    "name": "Máy tăm nước AquaJet",
                    "price": 850000,
                    "currency": "VND",
                },
                "source": {},
            }
        ],
        "total_chars": 42,
    }

    list_response = generator.direct_response(
        intent=Intent.PRODUCT_LIST,
        confidence=0.8,
        context=context,
    )
    detail_response = generator.direct_response(
        intent=Intent.PRODUCT_DETAIL,
        confidence=0.8,
        context=context,
    )

    assert "Mình tìm thấy 1 sản phẩm phù hợp" in list_response.result.text
    assert "có giá 850.000 VND" in detail_response.result.text


def test_chitchat_generation_uses_no_rag_llm_json():
    ollama = AsyncMock()
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "intent": "CHITCHAT",
                "confidence": 0.94,
                "answer_type": "chitchat",
                "entities": [],
                "result": {
                    "text": "Mình là trợ lý của phòng khám, sẵn sàng hỗ trợ bạn.",
                    "items": [],
                    "assets": [],
                    "sources": [],
                    "missing_assets": [],
                },
                "safety": {
                    "medical_disclaimer_required": False,
                    "needs_human_support": False,
                },
            }
        ),
        latency_ms=21,
        model="generation",
    )
    generator = GroundedGenerator(Settings(), ollama)

    response, metadata = asyncio.run(
        generator.generate_chitchat_with_retry(
            query="Bạn là ai?",
            confidence=0.94,
            session=None,
        )
    )

    assert response.intent == Intent.CHITCHAT
    assert response.answer_type == "chitchat"
    assert response.result.items == []
    assert metadata["llm"]["attempt_count"] == 1


def test_greeting_generation_uses_no_rag_llm_json():
    ollama = AsyncMock()
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "intent": "GREETING",
                "confidence": 0.94,
                "answer_type": "greeting",
                "entities": [],
                "result": {
                    "text": "Xin chào, mình có thể hỗ trợ bạn tra cứu thông tin phòng khám.",
                    "items": [],
                    "assets": [],
                    "sources": [],
                    "missing_assets": [],
                },
                "safety": {
                    "medical_disclaimer_required": False,
                    "needs_human_support": False,
                },
            }
        ),
        latency_ms=21,
        model="generation",
    )
    generator = GroundedGenerator(Settings(), ollama)

    response, metadata = asyncio.run(
        generator.generate_chitchat_with_retry(
            query="Xin chào",
            confidence=0.94,
            session=None,
            intent=Intent.GREETING,
        )
    )

    assert response.intent == Intent.GREETING
    assert response.answer_type == "greeting"
    assert response.result.items == []
    assert metadata["llm"]["attempt_count"] == 1


def test_synthesis_generation_uses_evidence_pack_json():
    ollama = AsyncMock()
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "answer": "Dịch vụ tẩy trắng răng có giá 1000000 VND.",
                "used_source_ids": ["service-1"],
                "medical_disclaimer_required": False,
                "needs_human_support": False,
            }
        ),
        latency_ms=32,
        model="generation",
    )
    context = {
        "items": [
            {
                "source_type": "service",
                "source_id": "service-1",
                "text": "Dịch vụ: Tẩy trắng răng. Giá: 1000000 VND",
                "raw_json": {"name": "Tẩy trắng răng", "price": 1000000},
                "source": {},
                "score": 1.0,
                "canonical_key": "service:service-1",
            }
        ],
        "total_chars": 48,
    }
    generator = GroundedGenerator(Settings(), ollama)

    response, metadata = asyncio.run(
        generator.generate_synthesis_with_retry(
            query="Tẩy trắng răng giá bao nhiêu?",
            intent=Intent.SERVICE_DETAIL,
            confidence=0.9,
            evidence_pack={"items": context["items"]},
            context=context,
            session=None,
        )
    )

    assert response.intent == Intent.SERVICE_DETAIL
    assert response.answer_type == "rag"
    assert response.result.items[0].id == "service-1"
    assert response.result.sources[0].source_id == "service-1"
    assert response.entities[0].matched_id == "service-1"
    assert metadata["llm"]["attempt_count"] == 1


def test_synthesis_requires_source_for_every_planned_task():
    ollama = AsyncMock()
    ollama.generate.return_value = OllamaResponse(
        text=json.dumps(
            {
                "answer": "Chỉ trả lời task dịch vụ.",
                "used_source_ids": ["service-1"],
            }
        ),
        latency_ms=10,
        model="generation",
    )
    context = {
        "items": [
            {
                "source_type": "service",
                "source_id": "service-1",
                "text": "Giá dịch vụ",
                "raw_json": {"name": "Dịch vụ"},
                "source": {"task_id": "t1"},
            },
            {
                "source_type": "faq",
                "source_id": "faq-1",
                "text": "FAQ an toàn",
                "raw_json": {"question": "Có đau không?", "answer": "Có thể ê nhẹ."},
                "source": {"task_id": "t2"},
            },
        ],
        "tasks": [
            {"task_id": "t1", "intent": "SERVICE_DETAIL"},
            {"task_id": "t2", "intent": "FAQ"},
        ],
    }

    with pytest.raises(Exception, match="task IDs"):
        asyncio.run(
            GroundedGenerator(Settings(), ollama).generate_synthesis_with_retry(
                query="Giá bao nhiêu và có đau không?",
                intent=Intent.SERVICE_DETAIL,
                confidence=0.9,
                evidence_pack={},
                context=context,
                session=None,
            )
        )
