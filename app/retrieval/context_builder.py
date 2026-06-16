from collections import Counter
from typing import Any

from app.observability.logging import get_logger
from app.retrieval.types import RetrievalResult

logger = get_logger(__name__)

# Priority order: structured/product/service results appear before
# unstructured chunks and table rows in the context window.
_SOURCE_PRIORITY: dict[str, int] = {
    "product": 0,
    "service": 1,
    "clinic_info": 2,
    "faq": 3,
    "table_row": 4,
    "chunk": 5,
}


class ContextBuilder:
    """Build a context payload from ranked retrieval results.

    Design rules:
    - Items sharing the same ``canonical_key`` are de-duplicated.
    - Items whose text exceeds ``max_item_chars`` are **skipped** rather than
      truncating or aborting the whole context window.
    - Per ``source_type`` caps prevent one collection from monopolising context.
    - Items are re-sorted by source priority to place structured data first.
    """

    def __init__(
        self,
        max_chars: int = 16_000,
        max_items_per_source: int = 4,
        max_item_chars: int = 4_000,
    ):
        self.max_chars = max_chars
        self.max_items_per_source = max_items_per_source
        self.max_item_chars = max_item_chars

    def build(
        self,
        results: list[RetrievalResult],
        *,
        apply_limits: bool = True,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        seen_canonical: set[str] = set()
        source_counts: Counter[str] = Counter()
        total_chars = 0
        skipped_long = 0
        skipped_dup = 0

        for result in results:
            text_len = len(result.text)

            # Skip oversized items instead of aborting the whole build.
            if apply_limits and text_len > self.max_item_chars:
                skipped_long += 1
                logger.debug(
                    "Skipping oversized context item %s (%d chars)",
                    result.key,
                    text_len,
                )
                continue

            # Deduplicate by normalised text content.
            normalized = " ".join(result.text.lower().split())
            if not normalized or normalized in seen:
                skipped_dup += 1
                continue

            # Deduplicate by canonical key (product:uuid, service:uuid, etc).
            if result.canonical_key and result.canonical_key in seen_canonical:
                skipped_dup += 1
                continue

            if (
                apply_limits
                and source_counts[result.source_type] >= self.max_items_per_source
            ):
                continue

            if apply_limits and total_chars + text_len > self.max_chars:
                # Budget exhausted – skip this item but continue to see if
                # smaller items from other source types still fit.
                continue

            seen.add(normalized)
            if result.canonical_key:
                seen_canonical.add(result.canonical_key)
            source_counts[result.source_type] += 1
            total_chars += text_len
            items.append(
                {
                    "source_type": result.source_type,
                    "source_id": result.source_id,
                    "text": result.text,
                    "raw_json": result.raw_json,
                    "source": result.source,
                    "score": result.score,
                    "canonical_key": result.canonical_key,
                }
            )

        # Re-sort by source priority so structured data comes first in the
        # LLM context window, improving grounding.
        items.sort(key=lambda item: _SOURCE_PRIORITY.get(item["source_type"], 99))

        return {
            "items": items,
            "total_chars": total_chars,
            "source_counts": dict(source_counts),
            "skipped_long": skipped_long,
            "skipped_dup": skipped_dup,
        }
