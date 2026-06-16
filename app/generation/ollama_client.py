import time
from dataclasses import dataclass

import httpx

from app.config import Settings


@dataclass
class OllamaResponse:
    text: str
    latency_ms: int
    model: str
    total_duration_ms: int = 0
    load_duration_ms: int = 0
    prompt_eval_duration_ms: int = 0
    eval_duration_ms: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    done_reason: str | None = None

    def trace_metadata(self) -> dict[str, int | str | None]:
        return {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "total_duration_ms": self.total_duration_ms,
            "load_duration_ms": self.load_duration_ms,
            "prompt_eval_duration_ms": self.prompt_eval_duration_ms,
            "eval_duration_ms": self.eval_duration_ms,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "done_reason": self.done_reason,
        }


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        system: str | None = None,
        json_mode: bool = False,
        timeout_seconds: int | None = None,
        think: bool = False,
    ) -> OllamaResponse:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": self.settings.ollama_num_predict,
            },
            "think": think,
            "keep_alive": self.settings.ollama_keep_alive,
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        started = time.perf_counter()
        timeout = self._timeout(timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed at {self.settings.ollama_base_url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        text = str(data.get("response") or data.get("message", {}).get("content") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return OllamaResponse(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=model,
            total_duration_ms=self._duration_ms(data.get("total_duration")),
            load_duration_ms=self._duration_ms(data.get("load_duration")),
            prompt_eval_duration_ms=self._duration_ms(data.get("prompt_eval_duration")),
            eval_duration_ms=self._duration_ms(data.get("eval_duration")),
            prompt_eval_count=int(data.get("prompt_eval_count") or 0),
            eval_count=int(data.get("eval_count") or 0),
            done_reason=data.get("done_reason"),
        )

    @staticmethod
    def _duration_ms(value) -> int:
        return int((int(value or 0)) / 1_000_000)

    def _timeout(self, timeout_seconds: int | None):
        value = (
            self.settings.ollama_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if value is None or value <= 0:
            return None
        return value
