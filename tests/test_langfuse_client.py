from contextlib import nullcontext
from unittest.mock import Mock

from app.config import Settings
from app.observability.langfuse_client import OptionalLangfuse


def test_langfuse_v3_trace_export_uses_observation_api():
    exporter = OptionalLangfuse(Settings(enable_langfuse=False))
    observation = Mock()
    client = Mock()
    client.start_as_current_observation.return_value = nullcontext(observation)
    exporter.client = client

    exporter.send_trace(
        id="internal-trace-id",
        name="simplydent-chat",
        session_id="session-1",
        input={"query": "Xin chào"},
        output={"answer": "Chào bạn"},
        metadata={"intent": "GREETING"},
    )

    client.start_as_current_observation.assert_called_once_with(
        name="simplydent-chat",
        as_type="chain",
        input={"query": "Xin chào"},
        output={"answer": "Chào bạn"},
        metadata={
            "intent": "GREETING",
            "internal_trace_id": "internal-trace-id",
        },
    )
    client.update_current_trace.assert_called_once_with(
        name="simplydent-chat",
        session_id="session-1",
        input={"query": "Xin chào"},
        output={"answer": "Chào bạn"},
        metadata={
            "intent": "GREETING",
            "internal_trace_id": "internal-trace-id",
        },
    )
