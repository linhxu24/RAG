from dataclasses import fields
from typing import Any, get_type_hints

from app.ner.entity_span_extractor import (
    EntitySpanExtractor,
    IngestionEntityExtractionResult,
    IngestionEntityMention,
)


def test_ingestion_time_ner_result_schema_contract():
    assert [field.name for field in fields(IngestionEntityMention)] == [
        "text",
        "label",
        "source_type",
        "source_index",
        "page_number",
        "start",
        "end",
        "score",
        "source",
        "metadata",
    ]
    assert get_type_hints(IngestionEntityMention) == {
        "text": str,
        "label": str,
        "source_type": str,
        "source_index": int,
        "page_number": int | None,
        "start": int | None,
        "end": int | None,
        "score": float,
        "source": str,
        "metadata": dict[str, Any],
    }
    assert [field.name for field in fields(IngestionEntityExtractionResult)] == [
        "provider",
        "mentions",
        "degraded",
        "error",
    ]
    assert get_type_hints(IngestionEntityExtractionResult) == {
        "provider": str,
        "mentions": list[IngestionEntityMention],
        "degraded": bool,
        "error": str | None,
    }


def test_ingestion_calls_ner_with_text_blocks_and_table_rows_contract():
    hints = get_type_hints(EntitySpanExtractor.extract_for_ingestion)
    assert hints["text_blocks"] == list[dict[str, Any]]
    assert hints["table_rows"] == list[dict[str, Any]] | None
    assert hints["known_products"] == list[str] | None
    assert hints["known_services"] == list[str] | None
    assert hints["return"] is IngestionEntityExtractionResult

    result = EntitySpanExtractor.__dict__["extract_for_ingestion"]
    assert result.__name__ == "extract_for_ingestion"
