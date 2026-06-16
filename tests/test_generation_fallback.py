import asyncio
import json
from unittest.mock import AsyncMock, Mock

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

    assert response.result.text.startswith("Trả lời đúng")
    assert len(response.result.items) == 1
    assert response.result.items[0].id == "faq-1"


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
