import asyncio
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.constants import Intent
from app.retrieval.query_rewrite import QueryRewriter


@pytest.mark.parametrize(
    "intent",
    [
        Intent.GREETING,
        Intent.CHITCHAT,
        Intent.CLINIC_INFO,
        Intent.PRODUCT_LIST,
        Intent.PRODUCT_DETAIL,
        Intent.PRODUCT_COMPARE,
        Intent.SERVICE_LIST,
        Intent.SERVICE_DETAIL,
        Intent.UNKNOWN,
    ],
)
def test_hyde_is_skipped_for_non_rag_intents(intent):
    ollama = AsyncMock()
    result = asyncio.run(
        QueryRewriter().rewrite(
            "Xin chào",
            intent,
            Settings(enable_hyde=True),
            ollama,
        )
    )

    assert result.hyde_used is False
    assert result.rewritten_query == "Xin chào"
    ollama.generate.assert_not_awaited()


def test_hyde_can_rewrite_faq_queries():
    ollama = AsyncMock()
    ollama.generate.return_value.text = "Đoạn truy xuất giả định"

    result = asyncio.run(
        QueryRewriter().rewrite(
            "Tại sao răng bị ê buốt?",
            Intent.FAQ,
            Settings(enable_hyde=True),
            ollama,
        )
    )

    assert result.hyde_used is True
    assert result.rewritten_query == "Đoạn truy xuất giả định"
    ollama.generate.assert_awaited_once()


def test_hyde_failure_preserves_original_query_and_reason():
    ollama = AsyncMock()
    ollama.generate.side_effect = TimeoutError("hyde timeout")

    result = asyncio.run(
        QueryRewriter().rewrite(
            "Tại sao răng bị ê buốt?",
            Intent.FAQ,
            Settings(enable_hyde=True),
            ollama,
        )
    )

    assert result.hyde_used is False
    assert result.rewritten_query == "Tại sao răng bị ê buốt?"
    assert result.failure_reason == "hyde timeout"
