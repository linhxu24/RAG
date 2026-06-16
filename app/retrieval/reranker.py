import time
from typing import Any

from app.config import Settings
from app.observability.logging import get_logger
from app.retrieval.types import RetrievalResult

logger = get_logger(__name__)


class OptionalReranker:
    """Cross-encoder reranker with lazy loading and graceful fallback.

    Tracks pre-rerank scores so callers can compute improvement metrics.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._attempted = False

    def rerank(
        self, query: str, results: list[RetrievalResult]
    ) -> tuple[list[RetrievalResult], bool, dict[str, Any]]:
        """Rerank *results* and return ``(reranked_list, was_applied, metadata)``.

        ``metadata`` always contains ``pre_rerank_top`` and ``post_rerank_top``
        so callers can log improvement rate even when reranking is disabled.
        """
        meta: dict[str, Any] = {
            "enabled": self.settings.enable_reranker,
            "input_count": len(results),
        }

        pre_top = results[0].key if results else None
        meta["pre_rerank_top"] = pre_top

        if not self.settings.enable_reranker or not results:
            truncated = results[: self.settings.final_top_k]
            meta["post_rerank_top"] = truncated[0].key if truncated else None
            meta["reranked"] = False
            return truncated, False, meta

        model = self._load()
        if model is None:
            truncated = results[: self.settings.final_top_k]
            meta["post_rerank_top"] = truncated[0].key if truncated else None
            meta["reranked"] = False
            meta["load_failed"] = True
            return truncated, False, meta

        top_n = results[: self.settings.rerank_top_n]
        pairs = [(query, item.text) for item in top_n]
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker inference failed; returning RRF order: %s", exc)
            truncated = results[: self.settings.final_top_k]
            meta["post_rerank_top"] = truncated[0].key if truncated else None
            meta["reranked"] = False
            meta["error"] = str(exc)
            return truncated, False, meta

        reranked = [
            RetrievalResult(
                source_type=item.source_type,
                source_id=item.source_id,
                text=item.text,
                score=float(score),
                raw_json=item.raw_json,
                source=item.source,
                ranks=item.ranks,
                canonical_key=item.canonical_key,
            )
            for item, score in zip(top_n, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        final = reranked[: self.settings.final_top_k]

        post_top = final[0].key if final else None
        meta["post_rerank_top"] = post_top
        meta["reranked"] = True
        meta["top_changed"] = pre_top != post_top
        meta["output_count"] = len(final)

        return final, True, meta

    def warmup(self) -> dict[str, Any]:
        started = time.perf_counter()
        already_loaded = self._model is not None
        model = self._load()
        return {
            "available": model is not None,
            "already_loaded": already_loaded,
            "load_latency_ms": int((time.perf_counter() - started) * 1000),
            "model": self.settings.reranker_model,
        }

    def _load(self):
        if self._attempted:
            return self._model
        self._attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.settings.reranker_model,
                device=self.settings.embedding_device,
            )
        except Exception as exc:
            logger.warning("Reranker unavailable; continuing without reranking: %s", exc)
        return self._model
