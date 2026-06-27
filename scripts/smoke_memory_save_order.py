"""Smoke-check chat finalization saves memory after response rendering."""

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from app.config import get_settings
from app.constants import Intent
from app.generation.renderer import ResponseRenderer
from app.generation.schemas import GeneratedResponse, ResultBody
from app.services.chat import ChatService


class RecordingTrace:
    def __init__(self) -> None:
        self.trace_id = uuid4()
        self.trace = SimpleNamespace(
            session_id="smoke-session",
            user_query="hello",
            total_latency_ms=0,
        )
        self.order: list[str] = []

    @contextmanager
    def step(self, step_name, input_data=None):
        self.order.append(step_name)
        state = {}
        yield state

    def finish(self, **kwargs):
        self.order.append("trace_finish")


class RecordingMemory:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def save_exchange(self, *args, **kwargs):
        self.order.append("memory_write")
        return {"saved": True, "summary": "ok"}


class NoopLangfuse:
    def send_trace(self, **payload):
        return None


def main() -> None:
    service = ChatService(get_settings().model_copy(update={"enable_langfuse": False}))
    trace = RecordingTrace()
    service.memory = RecordingMemory(trace.order)
    service.langfuse = NoopLangfuse()
    service.renderer = ResponseRenderer()
    response = GeneratedResponse(
        intent=Intent.GREETING,
        confidence=1.0,
        answer_type="greeting",
        result=ResultBody(text="Xin chào"),
    )
    rendered = service._finish(
        None,
        trace,
        response,
        False,
        {},
        1.0,
        memory_save_payload={
            "session_id": "smoke-session",
            "user_content": "hello",
            "assistant_content": "Xin chào",
            "detected_intents": ["GREETING"],
            "entities": {},
            "resolved_ids": {},
            "state": {},
            "trace_id": trace.trace_id,
            "suggestion_count": 0,
        },
    )
    assert rendered.answer.text == "Xin chào"
    assert trace.order == [
        "asset_resolver",
        "response_rendering",
        "memory_save",
        "memory_write",
        "trace_finish",
    ]
    print({"order": trace.order})


if __name__ == "__main__":
    main()
