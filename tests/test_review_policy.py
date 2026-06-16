import uuid

from app.db.models import Document
from app.ingestion.review import _approval_report
from app.ingestion.review_policy import split_review_reasons
from app.ingestion.smoke_checks import SmokeCheckReport


def test_review_reason_split_keeps_human_review_flags_waivable():
    result = split_review_reasons(
        [
            "review_required_by_upload_option",
            "table_1_classification_low_confidence",
            "document_type_requires_review",
            "one_or_more_image_references_were_not_resolved",
        ]
    )

    assert result.integrity_blockers == []
    assert len(result.review_only) == 4


def test_review_reason_split_recomputes_row_validation_instead_of_using_stale_reason():
    result = split_review_reasons(
        [
            "table_1_row_1_category_unrecognized",
            "one_or_more_embeddings_failed",
        ],
        ignore_dynamic_business_reasons=True,
    )

    assert result.integrity_blockers == ["one_or_more_embeddings_failed"]


def test_quality_report_split_keeps_current_row_validation_blockers():
    result = split_review_reasons(
        ["table_1_row_1_category_unrecognized"]
    )

    assert result.integrity_blockers == [
        "table_1_row_1_category_unrecognized"
    ]


def test_approval_report_merges_current_business_blockers(monkeypatch):
    document = Document(
        doc_id=uuid.uuid4(),
        file_name="services.csv",
        status="review_required",
        metadata_json={
            "review_reasons": [
                "review_required_by_upload_option",
                "table_1_row_1_category_unrecognized",
            ]
        },
    )
    smoke = SmokeCheckReport(True, {"retrievable_records": 1}, [], [])
    monkeypatch.setattr(
        "app.ingestion.review._current_business_blockers",
        lambda _session, _doc_id: ["table_1_row_1_category_unrecognized"],
    )

    report = _approval_report(object(), document, smoke)

    assert report.passed is False
    assert report.blocking_reasons == ["table_1_row_1_category_unrecognized"]
    assert report.checks["review_only_reasons"] == [
        "review_required_by_upload_option"
    ]


def test_approval_report_allows_review_only_reasons(monkeypatch):
    document = Document(
        doc_id=uuid.uuid4(),
        file_name="catalog.csv",
        status="review_required",
        metadata_json={
            "review_reasons": [
                "review_required_by_upload_option",
                "table_1_classification_low_confidence",
            ]
        },
    )
    smoke = SmokeCheckReport(True, {"retrievable_records": 1}, [], [])
    monkeypatch.setattr(
        "app.ingestion.review._current_business_blockers",
        lambda _session, _doc_id: [],
    )

    report = _approval_report(object(), document, smoke)

    assert report.passed is True
    assert report.blocking_reasons == []
