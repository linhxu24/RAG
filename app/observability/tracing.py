import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.constants import TRACE_STEPS
from app.db.models import RagTrace, RagTraceStep


def json_summary(value: Any, max_chars: int = 4_000) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        result = value
    elif isinstance(value, (list, tuple)):
        result = {"items": list(value)}
    else:
        result = {"value": str(value)}
    text = str(result)
    if len(text) <= max_chars:
        return result
    return {"summary": text[:max_chars], "truncated": True}


@dataclass
class TraceRecorder:
    session: Session
    trace: RagTrace
    started_at: float = field(default_factory=time.perf_counter)
    recorded_steps: set[str] = field(default_factory=set)

    @classmethod
    def start(cls, session: Session, user_query: str, session_id: str | None) -> "TraceRecorder":
        trace = RagTrace(user_query=user_query, session_id=session_id, status="running")
        session.add(trace)
        session.commit()
        session.refresh(trace)
        return cls(session=session, trace=trace)

    @property
    def trace_id(self) -> uuid.UUID:
        return self.trace.trace_id

    def record(
        self,
        step_name: str,
        *,
        input_data: Any = None,
        output_data: Any = None,
        latency_ms: int = 0,
        status: str = "success",
        error_message: str | None = None,
    ) -> RagTraceStep:
        step = RagTraceStep(
            trace_id=self.trace.trace_id,
            step_name=step_name,
            input_json=json_summary(input_data),
            output_json=json_summary(output_data),
            latency_ms=max(0, int(latency_ms)),
            status=status,
            error_message=error_message,
        )
        self.session.add(step)
        self.session.commit()
        self.recorded_steps.add(step_name)
        return step

    @contextmanager
    def step(self, step_name: str, input_data: Any = None) -> Iterator[dict[str, Any]]:
        start = time.perf_counter()
        state: dict[str, Any] = {"output": None}
        try:
            yield state
        except Exception as exc:
            self.record(
                step_name,
                input_data=input_data,
                output_data=state.get("output"),
                latency_ms=int((time.perf_counter() - start) * 1000),
                status="failed",
                error_message=str(exc),
            )
            raise
        else:
            self.record(
                step_name,
                input_data=input_data,
                output_data=state.get("output"),
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

    def skip(self, step_name: str, reason: str) -> None:
        self.record(step_name, output_data={"reason": reason}, status="skipped")

    def finish(
        self,
        *,
        intent: str,
        confidence: float,
        answer: dict[str, Any],
        status: str = "success",
    ) -> None:
        for step_name in TRACE_STEPS:
            if step_name not in self.recorded_steps:
                self.skip(step_name, "Step was not required for this route")
        self.trace.detected_intent = intent
        self.trace.confidence = confidence
        self.trace.total_latency_ms = int((time.perf_counter() - self.started_at) * 1000)
        self.trace.final_answer = answer
        self.trace.status = status
        self.session.add(self.trace)
        self.session.commit()


def create_trace_step(
    session: Session,
    trace_id: uuid.UUID,
    step_name: str,
    *,
    input_data: Any = None,
    output_data: Any = None,
    latency_ms: int = 0,
    status: str = "success",
    error_message: str | None = None,
) -> RagTraceStep:
    step = RagTraceStep(
        trace_id=trace_id,
        step_name=step_name,
        input_json=json_summary(input_data),
        output_json=json_summary(output_data),
        latency_ms=latency_ms,
        status=status,
        error_message=error_message,
    )
    session.add(step)
    session.commit()
    return step
