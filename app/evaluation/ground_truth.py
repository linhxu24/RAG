from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    FAQ,
    Asset,
    Chunk,
    ClinicInfo,
    Document,
    Product,
    Service,
    TableRow,
)


@dataclass(frozen=True)
class GroundTruthResolution:
    expected_ids: list[str]
    unresolved_keys: list[str]


def resolve_expected_ids(session: Session, case: dict[str, Any]) -> GroundTruthResolution:
    identifiers = {
        str(value)
        for field in (
            "expected_doc_ids",
            "expected_chunk_ids",
            "expected_row_ids",
            "expected_asset_ids",
        )
        for value in case.get(field, [])
    }
    unresolved: list[str] = []
    for key in case.get("expected_source_keys", []):
        resolved = _resolve_source_key(session, key)
        if resolved:
            identifiers.update(resolved)
        else:
            unresolved.append(key)
    return GroundTruthResolution(sorted(identifiers), unresolved)


def _resolve_source_key(session: Session, key: str) -> list[str]:
    kind, separator, value = key.partition(":")
    if not separator or not value:
        return []
    kind = kind.lower().strip()
    value = value.strip()
    if kind == "product":
        return _values(
            session,
            select(Product.product_id).where(
                Product.status == "active",
                Product.name.ilike(value),
            ),
        )
    if kind == "service":
        return _values(
            session,
            select(Service.service_id).where(
                Service.status == "active",
                Service.name.ilike(value),
            ),
        )
    if kind == "faq":
        return _values(
            session,
            select(FAQ.faq_id).where(
                FAQ.is_active.is_(True),
                FAQ.question.ilike(value),
            ),
        )
    if kind == "clinic_info":
        return _values(
            session,
            select(ClinicInfo.id).where(
                ClinicInfo.status == "active",
                ClinicInfo.key.ilike(value),
            ),
        )
    if kind == "asset":
        return _values(
            session,
            select(Asset.asset_id).where(
                Asset.status == "active",
                (Asset.asset_token == value) | (Asset.stable_asset_key == value),
            ),
        )
    if kind == "document":
        return _values(
            session,
            select(Document.doc_id).where(
                Document.status == "active",
                Document.checksum == value,
            ),
        )
    if kind == "chunk":
        checksum, _, index = value.rpartition(":")
        if not checksum or not index.isdigit():
            return []
        return _values(
            session,
            select(Chunk.chunk_id)
            .join(Document, Document.doc_id == Chunk.doc_id)
            .where(
                Document.status == "active",
                Document.checksum == checksum,
                Chunk.status == "active",
                Chunk.chunk_index == int(index),
            ),
        )
    if kind == "table_row":
        checksum, _, entity_name = value.partition(":")
        if not checksum or not entity_name:
            return []
        return _values(
            session,
            select(TableRow.row_id)
            .join(Document, Document.doc_id == TableRow.doc_id)
            .where(
                Document.status == "active",
                Document.checksum == checksum,
                TableRow.status == "active",
                TableRow.entity_name.ilike(entity_name),
            ),
        )
    return []


def _values(session: Session, statement) -> list[str]:
    return [str(value) for value in session.scalars(statement).all()]
