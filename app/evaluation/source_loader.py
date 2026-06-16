import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import FAQ, Asset, Chunk, ClinicInfo, Product, Service, TableRow


def load_source_text(session: Session, identifiers: list[str]) -> str:
    parts: list[str] = []
    for value in identifiers:
        try:
            identifier = uuid.UUID(value)
        except (TypeError, ValueError):
            continue
        if chunk := session.get(Chunk, identifier):
            parts.append(chunk.content)
            continue
        if row := session.get(TableRow, identifier):
            parts.append(row.row_text)
            continue
        if product := session.get(Product, identifier):
            parts.append(_business_text("Sản phẩm", product))
            continue
        if service := session.get(Service, identifier):
            parts.append(_business_text("Dịch vụ", service))
            continue
        if faq := session.get(FAQ, identifier):
            parts.append(f"Câu hỏi: {faq.question}\nTrả lời: {faq.answer}")
            continue
        if clinic := session.get(ClinicInfo, identifier):
            parts.append(f"{clinic.key}: {clinic.value}")
    return "\n\n".join(parts)


def expected_asset_ids_for_sources(session: Session, identifiers: list[str]) -> list[str]:
    asset_ids: set[str] = set()
    for value in identifiers:
        try:
            identifier = uuid.UUID(value)
        except (TypeError, ValueError):
            continue
        for model in (Product, Service):
            record = session.get(model, identifier)
            if record is not None and record.asset_id is not None:
                asset_ids.add(str(record.asset_id))
        asset = session.get(Asset, identifier)
        if asset is not None:
            asset_ids.add(str(asset.asset_id))
    return sorted(asset_ids)


def _business_text(label: str, record: Any) -> str:
    values = [
        f"{label}: {record.name}",
        f"Mô tả: {record.description}" if record.description else None,
        f"Giá: {record.price}" if record.price is not None else None,
    ]
    if hasattr(record, "quantity") and record.quantity is not None:
        values.append(f"Số lượng: {record.quantity}")
    if hasattr(record, "duration_minutes") and record.duration_minutes is not None:
        values.append(f"Thời lượng: {record.duration_minutes} phút")
    return ". ".join(value for value in values if value)
