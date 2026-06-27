from inspect import signature
from typing import Any, get_type_hints

from sqlalchemy.orm import Session

from app.generation.generator import GroundedGenerator
from app.generation.llm_client import LLMClient, LLMResponse
from app.observability.tracing import TraceRecorder


def test_generation_llm_metadata_contract_for_tracing():
    response = LLMResponse(
        text="{}",
        latency_ms=10,
        model="model-a",
        total_duration_ms=20,
        load_duration_ms=1,
        prompt_eval_duration_ms=2,
        eval_duration_ms=3,
        prompt_eval_count=4,
        eval_count=5,
        done_reason="stop",
    )
    assert response.trace_metadata() == {
        "model": "model-a",
        "latency_ms": 10,
        "total_duration_ms": 20,
        "load_duration_ms": 1,
        "prompt_eval_duration_ms": 2,
        "eval_duration_ms": 3,
        "prompt_eval_count": 4,
        "eval_count": 5,
        "done_reason": "stop",
    }


def test_generation_and_trace_recorder_method_contracts():
    init_hints = get_type_hints(GroundedGenerator.__init__)
    assert init_hints["llm"] is LLMClient

    record_hints = get_type_hints(TraceRecorder.record)
    assert record_hints["step_name"] is str
    assert record_hints["input_data"] is Any
    assert record_hints["output_data"] is Any
    assert record_hints["latency_ms"] is int
    assert record_hints["status"] is str
    assert record_hints["error_message"] == str | None
    assert list(signature(TraceRecorder.record).parameters) == [
        "self",
        "step_name",
        "input_data",
        "output_data",
        "latency_ms",
        "status",
        "error_message",
    ]

    start_hints = get_type_hints(TraceRecorder.start)
    assert start_hints["session"] is Session
    assert start_hints["user_query"] is str
    assert start_hints["session_id"] == str | None
    assert start_hints["return"] is TraceRecorder
