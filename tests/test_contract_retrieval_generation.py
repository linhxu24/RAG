from typing import Any, get_type_hints

from app.generation.schemas import ResultBody
from app.generation.validator import ResponseValidator
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.types import RetrievalResult


def test_context_builder_output_contract_for_generation():
    hints = get_type_hints(ContextBuilder.build)
    assert hints["results"] == list[RetrievalResult]
    assert hints["apply_limits"] is bool
    assert hints["return"] == dict[str, Any]

    context = ContextBuilder(max_chars=1000).build(
        [
            RetrievalResult(
                source_type="product",
                source_id="p1",
                text="Sản phẩm: AquaJet",
                score=1.0,
                raw_json={"name": "AquaJet", "asset_id": "asset-1"},
                source={"doc_id": "doc-1", "page_number": 1},
                canonical_key="product:p1",
            )
        ]
    )
    assert list(context) == [
        "items",
        "total_chars",
        "source_counts",
        "skipped_long",
        "skipped_dup",
    ]
    assert list(context["items"][0]) == [
        "source_type",
        "source_id",
        "text",
        "raw_json",
        "source",
        "score",
        "canonical_key",
    ]
    assert isinstance(context["items"], list)
    assert isinstance(context["total_chars"], int)


def test_generation_validator_consumes_retrieval_context_item_fields():
    context_item = {
        "source_type": "product",
        "source_id": "p1",
        "text": "Sản phẩm: AquaJet",
        "raw_json": {"name": "AquaJet"},
        "source": {"doc_id": "doc-1"},
        "score": 1.0,
        "canonical_key": "product:p1",
    }
    payload = {
        "intent": "PRODUCT_DETAIL",
        "confidence": 1.0,
        "answer_type": "rag",
        "entities": [{"type": "product", "name": "AquaJet", "matched_id": "p1"}],
        "result": {
            "text": "AquaJet",
            "items": [{"type": "product", "id": "p1", "doc_id": "doc-1"}],
            "sources": [{"source_type": "product", "source_id": "p1", "doc_id": "doc-1"}],
        },
    }
    response = ResponseValidator().validate(
        payload,
        context={"items": [context_item], "total_chars": 17},
        session=None,
    )
    assert isinstance(response.result, ResultBody)
