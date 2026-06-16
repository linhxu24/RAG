import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import (
    FAQ,
    Asset,
    Chunk,
    ClinicInfo,
    Document,
    ParsedTable,
    Product,
    Service,
    TableRow,
)
from app.ingestion.business_dedup import activate_business_records
from app.ingestion.business_validation import validate_business_rows
from app.ingestion.review_policy import split_review_reasons
from app.ingestion.smoke_checks import SmokeCheckReport, run_ingestion_smoke_checks
from app.ingestion.table_normalizer import TableClassification


class ApprovalValidationError(ValueError):
    def __init__(self, report: SmokeCheckReport):
        self.report = report
        super().__init__(
            "Document cannot be approved: " + ", ".join(report.blocking_reasons)
        )


def approve_document_records(session: Session, doc_id: uuid.UUID) -> SmokeCheckReport:
    document = session.get(Document, doc_id)
    if document is None:
        raise ValueError(f"Document not found: {doc_id}")
    replacement_ids = {
        uuid.UUID(value)
        for value in (document.metadata_json or {}).get("duplicate_document_ids", [])
    }
    smoke_report = run_ingestion_smoke_checks(
        session,
        doc_id,
        require_embeddings=True,
        ignored_duplicate_doc_ids=replacement_ids,
    )
    report = _approval_report(session, document, smoke_report)
    if not report.passed:
        raise ApprovalValidationError(report)
    for replacement_id in replacement_ids:
        apply_document_status(session, replacement_id, "archived")
    apply_document_status(session, doc_id, "active")
    session.commit()
    return report


def _approval_report(
    session: Session,
    document: Document,
    smoke_report: SmokeCheckReport,
) -> SmokeCheckReport:
    persisted_reasons = list(
        (document.metadata_json or {}).get("review_reasons", [])
    )
    reason_split = split_review_reasons(
        persisted_reasons,
        ignore_dynamic_business_reasons=True,
    )
    business_blockers = _current_business_blockers(session, document.doc_id)
    blockers = list(
        dict.fromkeys(
            [
                *smoke_report.blocking_reasons,
                *reason_split.integrity_blockers,
                *business_blockers,
            ]
        )
    )
    checks: dict[str, Any] = {
        **smoke_report.checks,
        "review_only_reasons": reason_split.review_only,
        "persisted_integrity_reasons": reason_split.integrity_blockers,
        "current_business_blockers": business_blockers,
    }
    return SmokeCheckReport(
        passed=not blockers,
        checks=checks,
        blocking_reasons=blockers,
        warnings=smoke_report.warnings,
    )


def _current_business_blockers(
    session: Session,
    doc_id: uuid.UUID,
) -> list[str]:
    tables = session.scalars(
        select(ParsedTable)
        .where(ParsedTable.doc_id == doc_id)
        .order_by(ParsedTable.table_id)
    ).all()
    blockers: list[str] = []
    for table_index, table in enumerate(tables, start=1):
        rows = session.scalars(
            select(TableRow)
            .where(TableRow.table_id == table.table_id)
            .order_by(TableRow.row_index)
        ).all()
        entity_types = {row.entity_type for row in rows if row.entity_type}
        if len(entity_types) != 1:
            continue
        entity_type = next(iter(entity_types))
        metadata = table.metadata_json or {}
        validation = validate_business_rows(
            session,
            [row.row_json for row in rows],
            TableClassification(
                entity_type=entity_type,
                confidence=float(metadata.get("classification_confidence") or 1.0),
                reasons=list(metadata.get("classification_reasons") or []),
                column_mapping=dict(metadata.get("column_mapping") or {}),
                requires_review=bool(metadata.get("requires_review")),
            ),
            table_index=table_index,
        )
        blockers.extend(validation.blocking_reasons)
    return list(dict.fromkeys(blockers))


def set_document_status(session: Session, doc_id: uuid.UUID, status: str) -> None:
    apply_document_status(session, doc_id, status)
    session.commit()


def apply_document_status(
    session: Session,
    doc_id: uuid.UUID,
    status: str,
) -> dict[str, int]:
    session.execute(update(Document).where(Document.doc_id == doc_id).values(status=status))
    for model in (
        Chunk,
        Asset,
        ParsedTable,
        TableRow,
    ):
        session.execute(update(model).where(_doc_column(model) == doc_id).values(status=status))
    if status == "active":
        return activate_business_records(session, doc_id).as_dict()
    for model in (Product, Service, ClinicInfo):
        values = {"status": status}
        if status == "archived" and model in (Product, Service):
            values["valid_to"] = func.now()
        session.execute(update(model).where(_doc_column(model) == doc_id).values(**values))
    session.execute(
        update(FAQ)
        .where(FAQ.source_doc_id == doc_id)
        .values(is_active=status == "active")
    )
    return {
        "products_superseded": 0,
        "services_superseded": 0,
        "faqs_superseded": 0,
        "clinic_info_superseded": 0,
        "current_document_duplicates_archived": 0,
    }


def _doc_column(model):
    if model in (Product, Service):
        return model.source_doc_id
    if model is ClinicInfo:
        return model.source_doc_id
    return model.doc_id
