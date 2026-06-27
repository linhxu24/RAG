from dataclasses import fields
from inspect import signature
from typing import Any, get_type_hints

from sqlalchemy.orm import Session

from app.orchestration.schemas import BoundTask, BoundTaskPlan, EvidenceItem
from app.orchestration.tool_executor import ToolExecutionResult, ToolExecutor
from app.retrieval.types import RetrievalResult


def test_retrieval_result_to_evidence_item_contract():
    result_hints = get_type_hints(RetrievalResult)
    assert [field.name for field in fields(RetrievalResult)] == [
        "source_type",
        "source_id",
        "text",
        "score",
        "raw_json",
        "source",
        "ranks",
        "canonical_key",
    ]
    assert result_hints == {
        "source_type": str,
        "source_id": str,
        "text": str,
        "score": float,
        "raw_json": dict[str, Any],
        "source": dict[str, Any],
        "ranks": dict[str, int],
        "canonical_key": str | None,
    }
    result = RetrievalResult(
        source_type="product",
        source_id="p1",
        text="Product",
        score=1.0,
        raw_json={"asset_id": "asset-1"},
        source={"doc_id": "doc-1"},
        canonical_key="product:p1",
    )
    evidence = EvidenceItem.from_retrieval(
        task_id="t1",
        result=result,
        trust_level="authoritative",
    )
    assert evidence.model_dump(mode="json") == {
        "task_id": "t1",
        "source_type": "product",
        "source_id": "p1",
        "text": "Product",
        "score": 1.0,
        "trust_level": "authoritative",
        "raw_json": {"asset_id": "asset-1"},
        "source": {"doc_id": "doc-1"},
        "canonical_key": "product:p1",
        "asset_ids": ["asset-1"],
    }


def test_tool_executor_consumes_bound_plan_and_returns_tool_result_contract():
    hints = get_type_hints(ToolExecutor.execute_many)
    assert hints["session"] is Session
    assert hints["plan"] is BoundTaskPlan
    assert hints["valid_task_ids"] == tuple[str, ...] | None
    assert hints["return"] is ToolExecutionResult

    params = signature(ToolExecutor._execute_tool).parameters
    assert list(params) == ["self", "session", "task", "tool_name"]
    assert get_type_hints(ToolExecutor._execute_tool)["task"] is BoundTask

    assert get_type_hints(ToolExecutionResult) == {
        "evidence": list[EvidenceItem],
        "tool_counts": dict[str, int],
        "errors": list[dict[str, str]],
        "reranker_runs": list[dict[str, Any]],
    }
