import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings


@dataclass
class LLMResponse:
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


class LLMClient(Protocol):
    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        system: str | None = None,
        json_mode: bool = False,
        timeout_seconds: int | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        think: bool = False,
    ) -> LLMResponse: ...


class OllamaLLMClient:
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
        num_predict: int | None = None,
        num_ctx: int | None = None,
        think: bool = False,
    ) -> LLMResponse:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": (
                    self.settings.ollama_num_predict
                    if num_predict is None
                    else num_predict
                ),
            },
            "think": think,
            "keep_alive": self.settings.ollama_keep_alive,
        }
        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        started = time.perf_counter()
        timeout = _timeout(
            self.settings.ollama_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
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
        return LLMResponse(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=model,
            total_duration_ms=_duration_ms(data.get("total_duration")),
            load_duration_ms=_duration_ms(data.get("load_duration")),
            prompt_eval_duration_ms=_duration_ms(data.get("prompt_eval_duration")),
            eval_duration_ms=_duration_ms(data.get("eval_duration")),
            prompt_eval_count=int(data.get("prompt_eval_count") or 0),
            eval_count=int(data.get("eval_count") or 0),
            done_reason=data.get("done_reason"),
        )


class OpenAILLMClient:
    """OpenAI-compatible chat completions client.

    The project keeps Ollama as a demo provider. Production can switch to this
    provider without changing the orchestration, retrieval, or generation code.
    """

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
        num_predict: int | None = None,
        num_ctx: int | None = None,
        think: bool = False,
    ) -> LLMResponse:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
        }
        max_tokens = num_predict or self.settings.openai_max_tokens
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if num_ctx is not None:
            # Chat Completions does not expose context-window sizing. Preserve
            # the argument for interface compatibility and ignore it here.
            pass
        if think:
            # OpenAI chat models used here do not need a provider-specific
            # thinking flag. Keep the argument provider-neutral.
            pass

        started = time.perf_counter()
        timeout = _timeout(
            self.settings.openai_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI request failed at {self.settings.openai_base_url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=str(data.get("model") or model),
            prompt_eval_count=int(usage.get("prompt_tokens") or 0),
            eval_count=int(usage.get("completion_tokens") or 0),
            total_duration_ms=int((time.perf_counter() - started) * 1000),
            done_reason=choice.get("finish_reason"),
        )


def build_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        return OpenAILLMClient(settings)
    if provider == "ollama":
        return OllamaLLMClient(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def _timeout(value: int | None):
    if value is None or value <= 0:
        return None
    return value


def _duration_ms(value) -> int:
    return int((int(value or 0)) / 1_000_000)
