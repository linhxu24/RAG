from types import SimpleNamespace

from app.evaluation.runner import _rankings, _step_payload
from app.observability.tracing import json_summary


def test_trace_step_payload_contract_consumed_by_evaluation():
    step = SimpleNamespace(
        step_name="context_builder",
        status="success",
        latency_ms=12,
        input_json={"count": 1},
        output_json={"source_ids": ["source-1"]},
        error_message=None,
    )
    assert _step_payload(step) == {
        "step_name": "context_builder",
        "status": "success",
        "latency_ms": 12,
        "input": {"count": 1},
        "output": {"source_ids": ["source-1"]},
        "error_message": None,
    }


def test_evaluation_ranking_contract_from_trace_outputs():
    steps = [
        {
            "step_name": "structured_retrieval",
            "output": {"results": [{"id": "structured-1"}]},
        },
        {
            "step_name": "rrf_fusion",
            "output": {"results": [{"id": "rrf-1"}]},
        },
        {
            "step_name": "reranker",
            "output": {"results": [{"id": "reranked-1"}]},
        },
        {
            "step_name": "context_builder",
            "output": {"source_ids": ["final-1"]},
        },
    ]
    rankings = _rankings(steps, {"answer": {"sources": []}})
    assert rankings == {
        "final": ["final-1"],
        "before_rerank": ["rrf-1"],
        "after_rerank": ["reranked-1"],
    }


def test_truncated_trace_summary_preserves_evaluation_fields():
    summary = json_summary(
        {
            "source_ids": ["source-1"],
            "results": [{"id": "source-1", "text": "x" * 1000}],
            "extra": "y" * 10_000,
        },
        max_chars=100,
    )
    assert summary["truncated"] is True
    assert summary["source_ids"] == ["source-1"]
    assert summary["results"][0]["id"] == "source-1"
