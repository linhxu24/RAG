import re
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db.models import FAQ, ClinicInfo, Product, Service


def normalize_business_key(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or "").lower())
    ascii_value = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", ascii_value)).strip()


def business_key(
    entity_type: str | None,
    canonical: dict[str, Any],
) -> tuple[str, str] | None:
    if entity_type == "product":
        value = canonical.get("name")
    elif entity_type == "service":
        value = canonical.get("service_name") or canonical.get("name")
    elif entity_type == "faq":
        if not canonical.get("answer"):
            return None
        value = canonical.get("question")
    elif entity_type == "clinic_info":
        if canonical.get("value") in (None, ""):
            return None
        value = canonical.get("key")
    else:
        return None
    normalized = normalize_business_key(value)
    return (entity_type, normalized) if normalized else None


@dataclass
class BusinessActivationReport:
    products_superseded: int = 0
    services_superseded: int = 0
    faqs_superseded: int = 0
    clinic_info_superseded: int = 0
    current_document_duplicates_archived: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def activate_business_records(
    session: Session,
    doc_id: uuid.UUID,
) -> BusinessActivationReport:
    """Activate one canonical business record per key and archive older versions."""
    now = datetime.now(UTC)
    report = BusinessActivationReport()
    report.products_superseded, product_duplicates = _activate_versioned(
        session,
        Product,
        Product.product_id,
        Product.name,
        Product.source_doc_id,
        doc_id,
        now,
    )
    report.services_superseded, service_duplicates = _activate_versioned(
        session,
        Service,
        Service.service_id,
        Service.name,
        Service.source_doc_id,
        doc_id,
        now,
    )
    report.faqs_superseded, faq_duplicates = _activate_faqs(session, doc_id)
    report.clinic_info_superseded, clinic_duplicates = _activate_clinic_info(
        session,
        doc_id,
    )
    report.current_document_duplicates_archived = (
        product_duplicates
        + service_duplicates
        + faq_duplicates
        + clinic_duplicates
    )
    return report


def _activate_versioned(
    session: Session,
    model,
    id_column,
    key_column,
    doc_column,
    doc_id: uuid.UUID,
    now: datetime,
) -> tuple[int, int]:
    current = list(
        session.scalars(
            select(model)
            .where(doc_column == doc_id)
            .order_by(id_column)
        ).all()
    )
    existing = list(
        session.scalars(
            select(model).where(
                doc_column != doc_id,
                model.status == "active",
            )
        ).all()
    )
    current_groups = _group_by_key(current, lambda item: getattr(item, key_column.key))
    existing_groups = _group_by_key(existing, lambda item: getattr(item, key_column.key))
    superseded = 0
    current_duplicates = 0
    for key, records in current_groups.items():
        winner, *duplicates = records
        for duplicate in duplicates:
            duplicate.status = "archived"
            duplicate.valid_to = now
            current_duplicates += 1
        older = existing_groups.get(key, [])
        for record in older:
            record.status = "archived"
            record.valid_to = now
            superseded += 1
        session.flush()
        winner.version = max(
            [int(winner.version or 1), *(int(item.version or 1) for item in older)],
            default=0,
        ) + (1 if older else 0)
        winner.status = "active"
        winner.valid_to = None
    return superseded, current_duplicates


def _activate_faqs(
    session: Session,
    doc_id: uuid.UUID,
) -> tuple[int, int]:
    current = list(
        session.scalars(
            select(FAQ)
            .where(FAQ.source_doc_id == doc_id)
            .order_by(FAQ.faq_id)
        ).all()
    )
    existing = list(
        session.scalars(
            select(FAQ).where(
                FAQ.is_active.is_(True),
                or_(
                    FAQ.source_doc_id != doc_id,
                    FAQ.source_doc_id.is_(None),
                ),
            )
        ).all()
    )
    return _activate_boolean_records(
        session,
        current,
        existing,
        key=lambda item: item.question,
        active_attribute="is_active",
    )


def _activate_clinic_info(
    session: Session,
    doc_id: uuid.UUID,
) -> tuple[int, int]:
    current = list(
        session.scalars(
            select(ClinicInfo)
            .where(ClinicInfo.source_doc_id == doc_id)
            .order_by(ClinicInfo.id)
        ).all()
    )
    existing = list(
        session.scalars(
            select(ClinicInfo).where(
                or_(
                    ClinicInfo.source_doc_id != doc_id,
                    ClinicInfo.source_doc_id.is_(None),
                ),
                ClinicInfo.status == "active",
            )
        ).all()
    )
    current_groups = _group_by_key(current, lambda item: item.key)
    existing_groups = _group_by_key(existing, lambda item: item.key)
    superseded = 0
    current_duplicates = 0
    for key, records in current_groups.items():
        winner, *duplicates = records
        for duplicate in duplicates:
            duplicate.status = "archived"
            current_duplicates += 1
        for record in existing_groups.get(key, []):
            record.status = "archived"
            superseded += 1
        session.flush()
        winner.status = "active"
    return superseded, current_duplicates


def _activate_boolean_records(
    session: Session,
    current: list[Any],
    existing: list[Any],
    *,
    key,
    active_attribute: str,
) -> tuple[int, int]:
    current_groups = _group_by_key(current, key)
    existing_groups = _group_by_key(existing, key)
    superseded = 0
    current_duplicates = 0
    for normalized, records in current_groups.items():
        winner, *duplicates = records
        for duplicate in duplicates:
            setattr(duplicate, active_attribute, False)
            current_duplicates += 1
        for record in existing_groups.get(normalized, []):
            setattr(record, active_attribute, False)
            superseded += 1
        # Flush deactivation before enabling the winner so the partial unique
        # index never observes two active rows for the same normalized key.
        session.flush()
        setattr(winner, active_attribute, True)
    return superseded, current_duplicates


def _group_by_key(records: list[Any], key) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        normalized = normalize_business_key(key(record))
        if normalized:
            grouped[normalized].append(record)
    return grouped
