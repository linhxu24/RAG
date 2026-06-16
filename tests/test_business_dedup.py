import uuid
from unittest.mock import Mock

from app.ingestion.business_dedup import (
    business_key,
    normalize_business_key,
)
from app.ingestion.table_normalizer import normalize_table
from app.ingestion.table_processor import TableProcessor


def test_business_key_is_case_and_accent_insensitive():
    assert normalize_business_key("  Tẩy Trắng Răng  ") == "tay trang rang"
    assert business_key(
        "service",
        {"service_name": "Tẩy Trắng Răng"},
    ) == ("service", "tay trang rang")


def test_faq_without_answer_is_not_reserved_as_dedup_key():
    assert business_key("faq", {"question": "Có đau không?"}) is None


def test_table_processor_skips_duplicate_business_rows_in_same_run():
    rows = [
        {"Tên sản phẩm": "Oral-B", "Giá": "850.000"},
        {"Tên sản phẩm": "oral b", "Giá": "850.000"},
    ]
    normalized = normalize_table(rows, "Bảng sản phẩm", "product")
    processor = TableProcessor()
    processor.sync_business_record = Mock(return_value="product")
    session = Mock()

    counts = processor.process_rows(
        session,
        table_id=uuid.uuid4(),
        doc_id=uuid.uuid4(),
        rows=normalized.rows,
        table_name="Bảng sản phẩm",
        status="review_required",
        embeddings=[None, None],
        classification=normalized.classification,
        seen_business_keys=set(),
    )

    assert counts.rows == 2
    assert counts.products == 1
    assert counts.duplicates_skipped == 1
    assert processor.sync_business_record.call_count == 1
