from dataclasses import fields
from typing import Any, get_type_hints

from app.config import Settings
from app.ner.entity_span_extractor import EntitySpan, SpanExtractionResult
from app.orchestration.binding_pipeline import TaskBindingPipeline
from app.orchestration.context_binder import ContextBinder
from app.orchestration.schemas import BindingDecision, BoundTask, TaskPlan


def test_query_time_ner_schema_consumed_by_orchestration_is_exact():
    span_hints = get_type_hints(EntitySpan)
    assert [field.name for field in fields(EntitySpan)] == [
        "text",
        "label",
        "start",
        "end",
        "score",
        "source",
        "metadata",
    ]
    assert span_hints == {
        "text": str,
        "label": str,
        "start": int | None,
        "end": int | None,
        "score": float,
        "source": str,
        "metadata": dict[str, Any],
    }
    result_hints = get_type_hints(SpanExtractionResult)
    assert [field.name for field in fields(SpanExtractionResult)] == [
        "query",
        "spans",
        "provider",
        "degraded",
        "error",
    ]
    assert result_hints == {
        "query": str,
        "spans": list[EntitySpan],
        "provider": str,
        "degraded": bool,
        "error": str | None,
    }
    payload = SpanExtractionResult(
        query="q",
        spans=[EntitySpan(text="AquaJet", label="product_name", start=0, end=7)],
    ).as_dict()
    assert list(payload) == ["query", "provider", "degraded", "error", "spans"]
    assert list(payload["spans"][0]) == [
        "text",
        "label",
        "start",
        "end",
        "score",
        "source",
        "metadata",
    ]


def test_orchestration_binder_accepts_span_extraction_result_contract():
    hints = get_type_hints(ContextBinder.__init__)
    assert hints["settings"] is Settings
    bind_hints = get_type_hints(ContextBinder.bind)
    assert bind_hints["plan"] is TaskPlan
    assert bind_hints["original_query"] is str
    assert bind_hints["history"] == dict[str, Any]
    assert bind_hints["span_result"] is SpanExtractionResult
    assert bind_hints["prior_bound_tasks"] == tuple[BoundTask, ...]

    pipeline_hints = get_type_hints(TaskBindingPipeline.run)
    assert pipeline_hints["span_result"] is SpanExtractionResult
    assert pipeline_hints["return"].__name__ == "BindingPipelineResult"
    assert get_type_hints(BindingDecision)["explicit_spans"] == tuple[dict[str, Any], ...]
