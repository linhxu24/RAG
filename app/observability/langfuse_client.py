from typing import Any

from app.config import Settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class OptionalLangfuse:
    def __init__(self, settings: Settings):
        self.client: Any | None = None
        if not settings.enable_langfuse:
            return
        if not all(
            [
                settings.langfuse_public_key,
                settings.langfuse_secret_key,
                settings.langfuse_host,
            ]
        ):
            logger.warning("Langfuse is enabled but credentials are incomplete")
            return
        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception as exc:
            logger.warning("Langfuse initialization failed: %s", exc)

    def send_trace(self, **payload: Any) -> None:
        if self.client is None:
            return
        try:
            metadata = dict(payload.get("metadata") or {})
            if payload.get("id"):
                metadata["internal_trace_id"] = str(payload["id"])
            name = str(payload.get("name") or "simplydent-trace")
            trace_input = payload.get("input")
            trace_output = payload.get("output")
            with self.client.start_as_current_observation(
                name=name,
                as_type="chain",
                input=trace_input,
                output=trace_output,
                metadata=metadata,
            ):
                self.client.update_current_trace(
                    name=name,
                    session_id=payload.get("session_id"),
                    input=trace_input,
                    output=trace_output,
                    metadata=metadata,
                )
        except Exception as exc:
            logger.warning("Langfuse trace export failed: %s", exc)
