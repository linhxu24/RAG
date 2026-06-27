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


@dataclass
class IngestionEntityMention:
    text: str
    label: str
    source_type: str
    source_index: int
    page_number: int | None = None
    start: int | None = None
    end: int | None = None
    score: float = 0.0
    source: str = "fallback"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "source_type": self.source_type,
            "source_index": self.source_index,
            "page_number": self.page_number,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class IngestionEntityExtractionResult:
    provider: str = "fallback"
    mentions: list[IngestionEntityMention] = field(default_factory=list)
    degraded: bool = False
    error: str | None = None

    @property
    def total_mentions(self) -> int:
        return len(self.mentions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "degraded": self.degraded,
            "error": self.error,
            "total_mentions": self.total_mentions,
            "labels": sorted({mention.label for mention in self.mentions}),
            "mentions": [mention.as_dict() for mention in self.mentions],
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

    def extract_for_ingestion(
        self,
        *,
        text_blocks: list[dict[str, Any]],
        table_rows: list[dict[str, Any]] | None = None,
        known_products: list[str] | None = None,
        known_services: list[str] | None = None,
    ) -> IngestionEntityExtractionResult:
        """Extract document-level entity mentions for ingestion metadata."""
        known_products = known_products or []
        known_services = known_services or []
        mentions: list[IngestionEntityMention] = []
        provider = "fallback"
        degraded = False
        error: str | None = None

        for index, block in enumerate(text_blocks):
            text = str(block.get("text") or "")
            if not text.strip():
                continue
            result = self.extract(
                text,
                known_products=known_products,
                known_services=known_services,
            )
            if result.provider != "fallback":
                provider = result.provider
            degraded = degraded or result.degraded
            error = error or result.error
            mentions.extend(
                IngestionEntityMention(
                    text=span.text,
                    label=span.label,
                    source_type="text_block",
                    source_index=index,
                    page_number=_int_or_none(block.get("page_number")),
                    start=span.start,
                    end=span.end,
                    score=span.score,
                    source=span.source,
                    metadata=span.metadata,
                )
                for span in result.spans
            )

        for index, row in enumerate(table_rows or []):
            mentions.extend(_table_row_mentions(row, index))

        return IngestionEntityExtractionResult(
            provider=provider,
            mentions=_dedupe_ingestion_mentions(mentions),
            degraded=degraded,
            error=error,
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
            surface = _best_surface_span(query, normalized_name)
            spans.append(
                EntitySpan(
                    text=surface[0] if surface else name,
                    label=label,
                    start=surface[1] if surface else None,
                    end=surface[2] if surface else None,
                    score=1.0,
                    source="catalog_match",
                    metadata={"catalog_name": name, "match_type": "full_name"},
                )
            )
            continue
        alias = _best_distinctive_alias(normalized_query, normalized_name)
        if alias:
            surface = _best_surface_span(query, alias)
            spans.append(
                EntitySpan(
                    text=surface[0] if surface else alias,
                    label=label,
                    start=surface[1] if surface else None,
                    end=surface[2] if surface else None,
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
    surface = _best_surface_span(query, normalized_phrase)
    return surface[0] if surface else None


def _best_surface_span(
    query: str,
    normalized_phrase: str,
) -> tuple[str, int, int] | None:
    normalized_words = normalized_phrase.split()
    if not normalized_words:
        return None
    original_matches = list(re.finditer(r"\S+", query))
    original_tokens = [match.group(0) for match in original_matches]
    for start in range(0, len(original_tokens)):
        max_end = min(
            len(original_tokens),
            start + len(normalized_words) + 3,
        )
        for end in range(start + 1, max_end + 1):
            candidate = " ".join(original_tokens[start:end]).strip(" ,.;:?!")
            if normalize_vietnamese(candidate) == normalized_phrase:
                start_offset = original_matches[start].start()
                end_offset = original_matches[end - 1].end()
                return query[start_offset:end_offset].strip(" ,.;:?!"), start_offset, end_offset
    return None


def _dedupe_spans(spans: list[EntitySpan]) -> list[EntitySpan]:
    deduped: dict[tuple[str, str], EntitySpan] = {}
    for span in spans:
        key = (normalize_vietnamese(span.text), span.label)
        existing = deduped.get(key)
        if existing is None or span.score > existing.score:
            deduped[key] = span
    return sorted(
        _remove_shadowed_catalog_aliases(list(deduped.values())),
        key=lambda item: (item.start is None, item.start or 0, -item.score),
    )


def _remove_shadowed_catalog_aliases(spans: list[EntitySpan]) -> list[EntitySpan]:
    full_name_spans = [
        span
        for span in spans
        if span.source == "catalog_match"
        and span.metadata.get("match_type") == "full_name"
        and span.start is not None
        and span.end is not None
    ]
    filtered: list[EntitySpan] = []
    for span in spans:
        if (
            span.source == "catalog_match"
            and span.metadata.get("match_type") == "distinctive_alias"
            and span.start is not None
            and span.end is not None
            and any(
                full.label == span.label
                and full.score >= span.score
                and full.start is not None
                and full.end is not None
                and full.start <= span.start
                and span.end <= full.end
                for full in full_name_spans
            )
        ):
            continue
        filtered.append(span)
    return filtered


def _table_row_mentions(
    row: dict[str, Any],
    index: int,
) -> list[IngestionEntityMention]:
    entity_type = str(row.get("_entity_type") or row.get("entity_type") or "").strip()
    name = _first_text(row, "name", "service_name", "product_name", "question", "key")
    mentions: list[IngestionEntityMention] = []
    if name:
        mentions.append(
            IngestionEntityMention(
                text=name,
                label=_name_label(entity_type),
                source_type="table_row",
                source_index=index,
                score=1.0,
                source="table_schema",
                metadata={"entity_type": entity_type or None, "field": "name"},
            )
        )
    category = _first_text(row, "category", "source_category", "category_code")
    if category:
        mentions.append(
            IngestionEntityMention(
                text=category,
                label=_category_label(entity_type),
                source_type="table_row",
                source_index=index,
                score=0.95,
                source="table_schema",
                metadata={"entity_type": entity_type or None, "field": "category"},
            )
        )
    brand = _first_text(row, "brand")
    if brand:
        mentions.append(
            IngestionEntityMention(
                text=brand,
                label="brand",
                source_type="table_row",
                source_index=index,
                score=0.95,
                source="table_schema",
                metadata={"field": "brand"},
            )
        )
    for field_name, label in (
        ("symptoms", "symptom"),
        ("indications", "treatment_concern"),
    ):
        for value in _list_text(row.get(field_name)):
            mentions.append(
                IngestionEntityMention(
                    text=value,
                    label=label,
                    source_type="table_row",
                    source_index=index,
                    score=0.85,
                    source="table_schema",
                    metadata={"field": field_name},
                )
            )
    return mentions


def _name_label(entity_type: str) -> str:
    if entity_type == "product":
        return "product_name"
    if entity_type == "service":
        return "service_name"
    if entity_type == "faq":
        return "faq_question"
    if entity_type == "clinic_info":
        return "clinic_info_key"
    return "business_entity"


def _category_label(entity_type: str) -> str:
    if entity_type == "product":
        return "product_category"
    if entity_type == "service":
        return "service_category"
    return "category"


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dedupe_ingestion_mentions(
    mentions: list[IngestionEntityMention],
) -> list[IngestionEntityMention]:
    deduped: dict[tuple[str, str, str, int], IngestionEntityMention] = {}
    for mention in mentions:
        key = (
            normalize_vietnamese(mention.text),
            mention.label,
            mention.source_type,
            mention.source_index,
        )
        existing = deduped.get(key)
        if existing is None or mention.score > existing.score:
            deduped[key] = mention
    return sorted(
        deduped.values(),
        key=lambda item: (
            item.source_type,
            item.source_index,
            item.start is None,
            item.start or 0,
            -item.score,
        ),
    )


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
