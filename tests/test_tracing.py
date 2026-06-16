import uuid

from app.observability.tracing import create_trace_step, json_summary


class FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, value):
        self.added.append(value)

    def commit(self):
        self.committed = True


def test_trace_step_creation():
    session = FakeSession()
    trace_id = uuid.uuid4()
    step = create_trace_step(
        session,
        trace_id,
        "router_intent",
        input_data={"query": "xin chào"},
        output_data={"intent": "GREETING"},
        latency_ms=4,
    )
    assert step.trace_id == trace_id
    assert step.latency_ms == 4
    assert session.committed is True


def test_trace_summary_truncates_large_payload():
    summary = json_summary({"text": "x" * 100}, max_chars=20)
    assert summary["truncated"] is True
