from dataclasses import dataclass

from app.config import Settings
from app.constants import Intent


@dataclass
class RewriteResult:
    original_query: str
    normalized_query: str
    hyde_query: str | None
    hyde_used: bool
    failure_reason: str | None = None

    @property
    def rewritten_query(self) -> str:
        return self.hyde_query or self.original_query

    def as_dict(self) -> dict[str, object]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "hyde_query": self.hyde_query,
            "hyde_used": self.hyde_used,
            "failure_reason": self.failure_reason,
        }


class QueryRewriter:
    HYDE_INTENTS = {Intent.FAQ}

    async def rewrite(
        self,
        query: str,
        intent: Intent,
        settings: Settings,
        ollama_client,
    ) -> RewriteResult:
        from app.retrieval.normalization import normalize_vietnamese

        normalized = normalize_vietnamese(query)
        if not settings.enable_hyde or intent not in self.HYDE_INTENTS:
            return RewriteResult(query, normalized, None, False)
        prompt = (
            "Write one short hypothetical Vietnamese clinic-data passage that would answer "
            f"this query. Do not invent numeric facts: {query}"
        )
        try:
            response = await ollama_client.generate(
                prompt=prompt,
                model=settings.ollama_generation_model,
                system="Return only the hypothetical retrieval passage.",
                timeout_seconds=settings.hyde_timeout_seconds,
                think=False,
            )
            rewritten = response.text.strip()
            return RewriteResult(
                query,
                normalized,
                rewritten or None,
                bool(rewritten),
            )
        except Exception as exc:
            return RewriteResult(query, normalized, None, False, str(exc))
