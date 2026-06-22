from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.retrieval.normalization import normalize_vietnamese, query_tokens

GENERIC_ENTITY_TOKENS = {
    "brush",
    "dental",
    "dich",
    "floss",
    "kham",
    "mouthwash",
    "nha",
    "paste",
    "phong",
    "product",
    "rang",
    "service",
    "toothbrush",
    "toothpaste",
    "treatment",
    "vu",
}


@dataclass
class EntitySpan:
    text: str
    label: str
    start: int | None = None
    end: int | None = None
    score: float = 0.0
    source: str = "fallback"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class SpanExtractionResult:
    query: str
    spans: list[EntitySpan] = field(default_factory=list)
    provider: str = "fallback"
    degraded: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "provider": self.provider,
            "degraded": self.degraded,
            "error": self.error,
            "spans": [span.as_dict() for span in self.spans],
        }


class EntitySpanExtractor:
    """Extract explicit user-mentioned spans.

    GLiNER is optional. When unavailable, the fallback still extracts spans
    from known catalog/service names and common dental constraints so the
    context binder can distinguish explicit user mentions from LLM-injected
    entities.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._gliner_model: Any | None = None
        self._gliner_load_error: str | None = None
        self.labels = [
            "service_name",
            "product_name",
            "service_category",
            "product_category",
            "brand",
            "symptom",
            "treatment_concern",
            "price_constraint",
            "duration_constraint",
            "stock_constraint",
            "sort_preference",
        ]

    def warmup(self) -> dict[str, Any]:
        if not self.settings.enable_gliner_ner:
            return {
                "enabled": False,
                "loaded": False,
                "model": self.settings.gliner_model,
                "device": self.settings.gliner_device,
                "error": None,
            }
        model = self._load_gliner()
        return {
            "enabled": True,
            "loaded": model is not None,
            "model": self.settings.gliner_model,
            "device": self.settings.gliner_device,
            "error": self._gliner_load_error,
        }

    def extract(
        self,
        query: str,
        *,
        known_products: list[str] | None = None,
        known_services: list[str] | None = None,
    ) -> SpanExtractionResult:
        known_products = known_products or []
        known_services = known_services or []
        if self.settings.enable_gliner_ner:
            result = self._extract_with_gliner(query)
            if result is not None:
                fallback = self._fallback_spans(
                    query,
                    known_products=known_products,
                    known_services=known_services,
                )
                return SpanExtractionResult(
                    query=query,
                    spans=_dedupe_spans([*result.spans, *fallback]),
                    provider=result.provider,
                    degraded=result.degraded,
                    error=result.error,
                )
        return SpanExtractionResult(
            query=query,
            spans=_dedupe_spans(
                self._fallback_spans(
                    query,
                    known_products=known_products,
                    known_services=known_services,
                )
            ),
            provider="fallback",
            degraded=bool(self.settings.enable_gliner_ner and self._gliner_load_error),
            error=self._gliner_load_error,
        )

    def _extract_with_gliner(self, query: str) -> SpanExtractionResult | None:
        model = self._load_gliner()
        if model is None:
            return None
        try:
            raw_spans = model.predict_entities(
                query,
                self.labels,
                threshold=self.settings.gliner_threshold,
            )
        except Exception as exc:  # pragma: no cover - depends on optional model runtime
            return SpanExtractionResult(
                query=query,
                provider="gliner",
                degraded=True,
                error=str(exc),
            )
        spans = []
        for item in raw_spans or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            label = str(item.get("label") or "").strip()
            if not text or not label:
                continue
            spans.append(
                EntitySpan(
                    text=text,
                    label=label,
                    start=_int_or_none(item.get("start")),
                    end=_int_or_none(item.get("end")),
                    score=float(item.get("score") or 0.0),
                    source="gliner",
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"text", "label"}
                    },
                )
            )
        return SpanExtractionResult(query=query, spans=spans, provider="gliner")

    def _load_gliner(self):
        if self._gliner_model is not None:
            return self._gliner_model
        if self._gliner_load_error is not None:
            return None
        try:
            from gliner import GLiNER  # type: ignore[import-not-found]

            model = GLiNER.from_pretrained(self.settings.gliner_model)
            if hasattr(model, "to"):
                model = model.to(self.settings.gliner_device)
            if hasattr(model, "eval"):
                model.eval()
            self._gliner_model = model
            return self._gliner_model
        except Exception as exc:  # pragma: no cover - optional dependency
            self._gliner_load_error = str(exc)
            return None

    def _fallback_spans(
        self,
        query: str,
        *,
        known_products: list[str],
        known_services: list[str],
    ) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        spans.extend(_known_name_spans(query, known_services, "service_name"))
        spans.extend(_known_name_spans(query, known_products, "product_name"))
        spans.extend(_constraint_spans(query))
        return spans


def _known_name_spans(
    query: str,
    names: list[str],
    label: str,
) -> list[EntitySpan]:
    normalized_query = normalize_vietnamese(query)
    if not normalized_query:
        return []
    spans: list[EntitySpan] = []
    for name in names:
        normalized_name = normalize_vietnamese(name)
        if not normalized_name:
            continue
        if normalized_name in normalized_query:
            spans.append(
                EntitySpan(
                    text=_best_surface(query, normalized_name) or name,
                    label=label,
                    score=1.0,
                    source="catalog_match",
                    metadata={"catalog_name": name, "match_type": "full_name"},
                )
            )
            continue
        alias = _best_distinctive_alias(normalized_query, normalized_name)
        if alias:
            spans.append(
                EntitySpan(
                    text=_best_surface(query, alias) or alias,
                    label=label,
                    score=0.82,
                    source="catalog_match",
                    metadata={"catalog_name": name, "match_type": "distinctive_alias"},
                )
            )
    return spans


def _constraint_spans(query: str) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    normalized = normalize_vietnamese(query)
    price_match = re.search(
        r"\b(?:duoi|tren|tu|den|khoang)?\s*\d+(?:[\.,]\d+)?\s*(?:k|nghin|trieu|vnd|dong)\b",
        normalized,
    )
    if price_match:
        spans.append(
            EntitySpan(
                text=price_match.group(0),
                label="price_constraint",
                score=0.95,
                source="regex",
            )
        )
    if any(term in normalized for term in ("mat bao lau", "bao lau", "thoi gian")):
        spans.append(
            EntitySpan(
                text="duration",
                label="duration_constraint",
                score=0.8,
                source="regex",
            )
        )
    if any(term in normalized for term in ("con hang", "ton kho", "so luong")):
        spans.append(
            EntitySpan(
                text="availability",
                label="stock_constraint",
                score=0.8,
                source="regex",
            )
        )
    if any(
        term in normalized
        for term in ("sap xep", "tang dan", "giam dan", "re nhat", "cao nhat")
    ):
        spans.append(
            EntitySpan(
                text="sort",
                label="sort_preference",
                score=0.8,
                source="regex",
            )
        )
    return spans


def _best_distinctive_alias(normalized_query: str, normalized_name: str) -> str | None:
    name_tokens = query_tokens(normalized_name)
    distinctive = [
        token
        for token in name_tokens
        if len(token) >= 5 and token not in GENERIC_ENTITY_TOKENS
    ]
    for length in range(min(4, len(name_tokens)), 0, -1):
        for start in range(0, len(name_tokens) - length + 1):
            phrase_tokens = name_tokens[start : start + length]
            phrase = " ".join(phrase_tokens)
            if phrase in normalized_query and (
                (length >= 2 and any(token in distinctive for token in phrase_tokens))
                or phrase_tokens[0] in distinctive
            ):
                return phrase
    return None


def _best_surface(query: str, normalized_phrase: str) -> str | None:
    normalized_words = normalized_phrase.split()
    if not normalized_words:
        return None
    original_tokens = re.findall(r"\S+", query)
    for start in range(0, len(original_tokens)):
        max_end = min(
            len(original_tokens),
            start + len(normalized_words) + 3,
        )
        for end in range(start + 1, max_end + 1):
            candidate = " ".join(original_tokens[start:end]).strip(" ,.;:?!")
            if normalize_vietnamese(candidate) == normalized_phrase:
                return candidate
    return None


def _dedupe_spans(spans: list[EntitySpan]) -> list[EntitySpan]:
    deduped: dict[tuple[str, str], EntitySpan] = {}
    for span in spans:
        key = (normalize_vietnamese(span.text), span.label)
        existing = deduped.get(key)
        if existing is None or span.score > existing.score:
            deduped[key] = span
    return sorted(
        deduped.values(),
        key=lambda item: (item.start is None, item.start or 0, -item.score),
    )


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
