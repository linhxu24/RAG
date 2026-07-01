import asyncio
import json

from app.generation.llm_client import LLMResponse
from app.orchestration.query_rewriter import (
    SYSTEM_PROMPT,
    QueryRewriteInput,
    RawTurn,
    RecentEntity,
    rewrite_query,
)


class _RewriteLLM:
    def __init__(self, text: str):
        self.text = text
        self.kwargs = {}

    async def generate(self, **kwargs):
        self.kwargs = kwargs
        return LLMResponse(
            text=self.text,
            latency_ms=12,
            model=str(kwargs.get("model") or "rewrite-model"),
        )


def test_contextual_query_rewriter_uses_few_shot_prompt_shape():
    llm = _RewriteLLM(
        json.dumps(
            {
                "rewritten_query": "SilkLine Waxed Dental Floss giá bao nhiêu?",
                "is_standalone": False,
                "needs_clarification": False,
                "referenced_entities": ["SilkLine Waxed Dental Floss"],
            },
            ensure_ascii=False,
        )
    )
    payload = QueryRewriteInput(
        current_query="Còn loại kia thì sao?",
        turns=[
            RawTurn(
                role="user",
                text=(
                    "AquaJet Mini Water Flosser giá bao nhiêu?"
                ),
            ),
            RawTurn(
                role="assistant",
                text="AquaJet Mini Water Flosser có giá 1.250.000 VND.",
            ),
        ],
        recent_entities=[
            RecentEntity(name="AquaJet Mini Water Flosser", type="product"),
            RecentEntity(name="SilkLine Waxed Dental Floss", type="product"),
        ],
    )

    result = asyncio.run(
        rewrite_query(llm, payload, model="rewrite-model", timeout_s=8.0)
    )

    assert result.rewritten_query == "SilkLine Waxed Dental Floss giá bao nhiêu?"
    assert result.is_standalone is False
    assert result.needs_clarification is False
    assert result.referenced_entities == ["SilkLine Waxed Dental Floss"]
    assert llm.kwargs["system"] == SYSTEM_PROMPT
    assert llm.kwargs["model"] == "rewrite-model"
    assert llm.kwargs["json_mode"] is True
    assert llm.kwargs["timeout_seconds"] == 8
    assert "### Ví dụ 4" in llm.kwargs["prompt"]
    assert 'Câu hỏi hiện tại: "Còn loại kia thì sao?"' in llm.kwargs["prompt"]


def test_contextual_query_rewriter_falls_back_to_original_query_on_llm_error():
    class FailingLLM:
        async def generate(self, **_kwargs):
            raise TimeoutError("rewrite timeout")

    payload = QueryRewriteInput(
        current_query="Mất bao lâu?",
        turns=[RawTurn(role="user", text="Tẩy trắng răng giá bao nhiêu?")],
        recent_entities=[
            RecentEntity(name="Tẩy trắng răng tại phòng khám", type="service")
        ],
    )

    result = asyncio.run(
        rewrite_query(FailingLLM(), payload, model="rewrite-model", timeout_s=1.0)
    )

    assert result.rewritten_query == "Mất bao lâu?"
    assert result.is_standalone is True
    assert result.needs_clarification is False
    assert result.referenced_entities == []


def test_contextual_query_rewriter_repairs_single_entity_context_miss():
    llm = _RewriteLLM(
        json.dumps(
            {
                "rewritten_query": "Còn hàng không?",
                "is_standalone": True,
                "needs_clarification": False,
                "referenced_entities": [],
            },
            ensure_ascii=False,
        )
    )
    payload = QueryRewriteInput(
        current_query="Còn hàng không?",
        turns=[
            RawTurn(
                role="user",
                text="AquaJet Mini Water Flosser giá bao nhiêu?",
            ),
            RawTurn(
                role="assistant",
                text="AquaJet Mini Water Flosser có giá 1.250.000 VND.",
            ),
        ],
        recent_entities=[
            RecentEntity(name="AquaJet Mini Water Flosser", type="product")
        ],
    )

    result = asyncio.run(
        rewrite_query(llm, payload, model="rewrite-model", timeout_s=8.0)
    )

    assert result.rewritten_query == "AquaJet Mini Water Flosser: Còn hàng không?"
    assert result.is_standalone is False
    assert result.referenced_entities == ["AquaJet Mini Water Flosser"]


def test_contextual_query_rewriter_does_not_repair_independent_schedule_query():
    llm = _RewriteLLM(
        json.dumps(
            {
                "rewritten_query": "Phòng khám mở cửa mấy giờ?",
                "is_standalone": True,
                "needs_clarification": False,
                "referenced_entities": [],
            },
            ensure_ascii=False,
        )
    )
    payload = QueryRewriteInput(
        current_query="Phòng khám mở cửa mấy giờ?",
        turns=[
            RawTurn(
                role="user",
                text="AquaJet Mini Water Flosser giá bao nhiêu?",
            ),
            RawTurn(
                role="assistant",
                text="AquaJet Mini Water Flosser có giá 1.250.000 VND.",
            ),
        ],
        recent_entities=[
            RecentEntity(name="AquaJet Mini Water Flosser", type="product")
        ],
    )

    result = asyncio.run(
        rewrite_query(llm, payload, model="rewrite-model", timeout_s=8.0)
    )

    assert result.rewritten_query == "Phòng khám mở cửa mấy giờ?"
    assert result.is_standalone is True
    assert result.referenced_entities == []
