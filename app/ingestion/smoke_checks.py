import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.assets.resolver import detect_asset_tokens
from app.db.models import (
    FAQ,
    Asset,
    Chunk,
    ChunkAsset,
    Document,
    ParsedTable,
    Product,
    Service,
    TableRow,
)


@dataclass(frozen=True)
class SmokeCheckReport:
    passed: bool
    checks: dict[str, Any]
    blocking_reasons: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
        }


def run_ingestion_smoke_checks(
    session: Session,
    doc_id: uuid.UUID,
    *,
    require_embeddings: bool,
    staged_asset_paths: dict[uuid.UUID, str] | None = None,
    ignored_duplicate_doc_ids: set[uuid.UUID] | None = None,
) -> SmokeCheckReport:
    document = session.get(Document, doc_id)
    if document is None:
        return SmokeCheckReport(False, {}, ["document_missing"], [])

    chunks = session.scalars(select(Chunk).where(Chunk.doc_id == doc_id)).all()
    tables = session.scalars(select(ParsedTable).where(ParsedTable.doc_id == doc_id)).all()
    rows = session.scalars(select(TableRow).where(TableRow.doc_id == doc_id)).all()
    assets = session.scalars(select(Asset).where(Asset.doc_id == doc_id)).all()
    chunk_assets = session.scalars(
        select(ChunkAsset)
        .join(Asset, Asset.asset_id == ChunkAsset.asset_id)
        .where(Asset.doc_id == doc_id)
    ).all()
    faq_count = session.scalar(
        select(func.count())
        .select_from(FAQ)
        .where(FAQ.source_doc_id == doc_id)
    ) or 0

    blocking: list[str] = []
    warnings: list[str] = []
    retrievable_records = len(chunks) + len(rows) + faq_count
    if retrievable_records == 0:
        blocking.append("no_retrievable_records")

    missing_chunk_embeddings = sum(chunk.embedding is None for chunk in chunks)
    missing_row_embeddings = sum(row.embedding is None for row in rows)
    if require_embeddings and (missing_chunk_embeddings or missing_row_embeddings):
        blocking.append("retrievable_records_missing_embeddings")

    expected_table_rows = sum(
        len(table.table_json) if isinstance(table.table_json, list) else 0 for table in tables
    )
    if expected_table_rows != len(rows):
        blocking.append("table_row_count_mismatch")

    unclassified_rows = sum(row.entity_type is None for row in rows)
    if unclassified_rows:
        blocking.append("unclassified_table_rows")

    linked_asset_ids = {link.asset_id for link in chunk_assets}
    linked_asset_ids.update(
        session.scalars(
            select(Product.asset_id).where(
                Product.source_doc_id == doc_id,
                Product.asset_id.is_not(None),
            )
        ).all()
    )
    linked_asset_ids.update(
        session.scalars(
            select(Service.asset_id).where(
                Service.source_doc_id == doc_id,
                Service.asset_id.is_not(None),
            )
        ).all()
    )
    unlinked_assets = [asset for asset in assets if asset.asset_id not in linked_asset_ids]
    if unlinked_assets:
        blocking.append("assets_not_linked_to_content")

    all_chunk_tokens = {
        token for chunk in chunks for token in detect_asset_tokens(chunk.content)
    }
    persisted_tokens = {asset.asset_token for asset in assets}
    missing_tokens = sorted(all_chunk_tokens - persisted_tokens)
    if missing_tokens:
        blocking.append("chunk_contains_unresolved_asset_tokens")

    staged_asset_paths = staged_asset_paths or {}
    broken_asset_files = []
    for asset in assets:
        candidate = staged_asset_paths.get(asset.asset_id) or asset.local_path
        if not candidate or not Path(candidate).is_file():
            broken_asset_files.append(str(asset.asset_id))
    if broken_asset_files:
        blocking.append("asset_files_missing")

    orphan_business_records = (
        session.scalar(
            select(func.count())
            .select_from(Product)
            .where(Product.source_doc_id == doc_id, Product.source_row_id.is_(None))
        )
        or 0
    ) + (
        session.scalar(
            select(func.count())
            .select_from(Service)
            .where(Service.source_doc_id == doc_id, Service.source_row_id.is_(None))
        )
        or 0
    )
    if orphan_business_records:
        blocking.append("business_records_missing_source_rows")

    duplicate_conditions = [
        Document.doc_id != doc_id,
        Document.checksum == document.checksum,
        Document.status == "active",
    ]
    if ignored_duplicate_doc_ids:
        duplicate_conditions.append(Document.doc_id.not_in(ignored_duplicate_doc_ids))
    active_duplicate_count = session.scalar(
        select(func.count())
        .select_from(Document)
        .where(*duplicate_conditions)
    ) or 0
    if active_duplicate_count:
        blocking.append("duplicate_active_document_checksum")

    checks = {
        "retrievable_records": retrievable_records,
        "chunks": len(chunks),
        "table_rows": len(rows),
        "faqs": faq_count,
        "missing_chunk_embeddings": missing_chunk_embeddings,
        "missing_table_row_embeddings": missing_row_embeddings,
        "expected_table_rows": expected_table_rows,
        "actual_table_rows": len(rows),
        "unclassified_table_rows": unclassified_rows,
        "assets": len(assets),
        "chunk_asset_links": len(chunk_assets),
        "unlinked_asset_ids": [str(asset.asset_id) for asset in unlinked_assets],
        "unresolved_asset_tokens": missing_tokens,
        "broken_asset_ids": broken_asset_files,
        "orphan_business_records": orphan_business_records,
        "active_duplicate_count": active_duplicate_count,
    }
    return SmokeCheckReport(not blocking, checks, list(dict.fromkeys(blocking)), warnings)
