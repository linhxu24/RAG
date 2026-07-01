import asyncio
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings
from app.constants import TRACE_STEPS, Intent
from app.generation.schemas import ChatRequest
from app.observability.tracing import TraceRecorder
from app.services.chat import ChatService


class FakeTraceRecorder:
    def __init__(self):
        self.trace_id = uuid.uuid4()
        self.trace = SimpleNamespace(
            session_id=None,
            user_query="",
            total_latency_ms=0,
        )
        self.recorded_steps = set()
        self.records = []
        self.finished = None

    def record(
        self,
        step_name,
        *,
        input_data=None,
        output_data=None,
        latency_ms=0,
        status="success",
        error_message=None,
    ):
        self.recorded_steps.add(step_name)
        self.records.append(
            {
                "step_name": step_name,
                "input_data": input_data,
                "output_data": output_data,
                "latency_ms": latency_ms,
                "status": status,
                "error_message": error_message,
            }
        )

    @contextmanager
    def step(self, step_name, input_data=None):
        state = {"output": None}
        yield state
        self.record(step_name, input_data=input_data, output_data=state.get("output"))

    def skip(self, step_name, reason):
        self.record(step_name, output_data={"reason": reason}, status="skipped")

    def finish(self, *, intent, confidence, answer, status="success"):
        self.finished = {
            "intent": intent,
            "confidence": confidence,
            "answer": answer,
            "status": status,
        }


def _chat_service(settings: Settings) -> ChatService:
    service = ChatService(settings)
    service.langfuse = SimpleNamespace(send_trace=lambda **_: None)
    return service


def test_contextual_query_rewrite_trace_step_order():
    assert TRACE_STEPS.index("contextual_query_rewrite") == (
        TRACE_STEPS.index("memory_load") + 1
    )
    assert TRACE_STEPS.index("contextual_query_rewrite") < TRACE_STEPS.index(
        "task_planning"
    )


def test_evidence_first_runtime_path_is_recorded(monkeypatch):
    fake_trace = FakeTraceRecorder()
    monkeypatch.setattr(
        TraceRecorder,
        "start",
        classmethod(lambda cls, session, user_query, session_id: fake_trace),
    )
    service = _chat_service(Settings(enable_multi_task_planner=True))

    async def fail_with_recoverable_error(*_args, **_kwargs):
        raise RuntimeError("provider timeout")

    monkeypatch.setattr(
        service,
        "_chat_with_evidence_pipeline",
        fail_with_recoverable_error,
    )

    response = asyncio.run(
        service.chat(SimpleNamespace(), ChatRequest(message="hello", debug=True))
    )

    runtime_record = next(
        record for record in fake_trace.records if record["step_name"] == "runtime_path"
    )
    assert runtime_record["output_data"]["active"] == ChatService.EVIDENCE_FIRST_RUNTIME
    assert response.intent == Intent.UNKNOWN
    assert response.degraded is True
    assert response.debug["runtime_path"] == ChatService.EVIDENCE_FIRST_RUNTIME
    assert fake_trace.finished["status"] == "degraded"


def test_legacy_runtime_is_labeled_debug_only(monkeypatch):
    fake_trace = FakeTraceRecorder()
    monkeypatch.setattr(
        TraceRecorder,
        "start",
        classmethod(lambda cls, session, user_query, session_id: fake_trace),
    )
    service = _chat_service(Settings(enable_multi_task_planner=False))

    def fail_before_legacy_retrieval(*_args, **_kwargs):
        raise RuntimeError("legacy provider unavailable")

    monkeypatch.setattr(service.structured, "active_names", fail_before_legacy_retrieval)

    response = asyncio.run(
        service.chat(SimpleNamespace(), ChatRequest(message="hello", debug=True))
    )

    assert response.degraded is True
    assert response.debug["runtime_path"] == ChatService.LEGACY_RUNTIME
    assert any(
        record["step_name"] == "legacy_runtime_warning"
        for record in fake_trace.records
    )


def test_database_failures_remain_failed_and_visible(monkeypatch):
    fake_trace = FakeTraceRecorder()
    monkeypatch.setattr(
        TraceRecorder,
        "start",
        classmethod(lambda cls, session, user_query, session_id: fake_trace),
    )
    service = _chat_service(Settings(enable_multi_task_planner=True))

    async def fail_with_database_error(*_args, **_kwargs):
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(
        service,
        "_chat_with_evidence_pipeline",
        fail_with_database_error,
    )

    with pytest.raises(RuntimeError, match="Chat pipeline failed"):
        asyncio.run(service.chat(SimpleNamespace(), ChatRequest(message="hello")))

    assert fake_trace.finished["status"] == "failed"
