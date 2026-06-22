import re
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets.resolver import detect_asset_tokens
from app.db.models import (
    FAQ,
    Asset,
    ClinicInfo,
    FAQAlias,
    Product,
    ProductAlias,
    Service,
    ServiceAlias,
    TableRow,
)
from app.ingestion.business_dedup import business_key
from app.ingestion.table_normalizer import (
    TableClassification,
    canonicalize_row,
    classify_table,
)
from app.taxonomy import normalize_label, resolve_category


def detect_entity_type(rows: list[dict[str, Any]], table_name: str | None = None) -> str | None:
    return classify_table(rows, table_name).entity_type


def serialize_row(row: dict[str, Any]) -> str:
    return " | ".join(f"{key}: {value}" for key, value in row.items() if value not in (None, ""))


def parse_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    cleaned = re.sub(r"[^\d,.-]", "", str(value)).strip()
    if not cleaned:
        return None
    sign = "-" if cleaned.startswith("-") else ""
    cleaned = cleaned.lstrip("+-")
    separators = [index for index, char in enumerate(cleaned) if char in ".,"]
    if separators:
        last_separator = separators[-1]
        decimal_digits = len(cleaned) - last_separator - 1
        groups = re.split(r"[.,]", cleaned)
        uses_thousands = decimal_digits == 3 and all(
            len(group) == 3 for group in groups[1:] if group
        )
        if uses_thousands:
            cleaned = "".join(groups)
        elif decimal_digits <= 2:
            integer = re.sub(r"[.,]", "", cleaned[:last_separator])
            fraction = cleaned[last_separator + 1 :]
            cleaned = f"{integer}.{fraction}"
        else:
            cleaned = re.sub(r"[.,]", "", cleaned)
    cleaned = sign + cleaned
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_int(value: Any) -> int | None:
    decimal = parse_decimal(value)
    return int(decimal) if decimal is not None else None


@dataclass
class TableSyncCounts:
    rows: int = 0
    products: int = 0
    services: int = 0
    faqs: int = 0
    clinic_info: int = 0
    duplicates_skipped: int = 0


class TableProcessor:
    def process_rows(
        self,
        session: Session,
        *,
        table_id: uuid.UUID,
        doc_id: uuid.UUID,
        rows: list[dict[str, Any]],
        table_name: str | None,
        status: str,
        embeddings: list[list[float] | None],
        classification: TableClassification | None = None,
        seen_business_keys: set[tuple[str, str]] | None = None,
    ) -> TableSyncCounts:
        classification = classification or classify_table(rows, table_name)
        entity_type = classification.entity_type
        counts = TableSyncCounts()
        seen_business_keys = seen_business_keys if seen_business_keys is not None else set()
        for index, raw_row in enumerate(rows):
            canonical = canonicalize_row(raw_row)
            entity_name = (
                str(canonical.get("service_name") or canonical.get("name") or "").strip() or None
            )
            table_row = TableRow(
                table_id=table_id,
                doc_id=doc_id,
                row_index=index,
                entity_type=entity_type,
                entity_name=entity_name,
                row_text=serialize_row(raw_row),
                row_json=raw_row,
                embedding=embeddings[index],
                status=status,
                metadata_json={"classification": classification.as_metadata()},
            )
            session.add(table_row)
            session.flush()
            counts.rows += 1
            dedup_key = business_key(entity_type, canonical)
            if dedup_key and dedup_key in seen_business_keys:
                table_row.metadata_json = {
                    **(table_row.metadata_json or {}),
                    "business_dedup": {
                        "key": f"{dedup_key[0]}:{dedup_key[1]}",
                        "action": "business_sync_skipped",
                    },
                }
                counts.duplicates_skipped += 1
                continue
            if dedup_key:
                seen_business_keys.add(dedup_key)
            synced_type = self.sync_business_record(
                session,
                table_row=table_row,
                entity_type=entity_type,
                status=status,
                canonical=canonical,
                embedding=embeddings[index],
            )
            if synced_type == "product":
                counts.products += 1
            elif synced_type == "service":
                counts.services += 1
            elif synced_type == "faq":
                counts.faqs += 1
            elif synced_type == "clinic_info":
                counts.clinic_info += 1
        return counts

    def sync_business_record(
        self,
        session: Session,
        *,
        table_row: TableRow,
        entity_type: str | None,
        status: str,
        canonical: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str | None:
        canonical = canonical or canonicalize_row(table_row.row_json)
        entity_name = str(
            canonical.get("service_name") or canonical.get("name") or ""
        ).strip()
        if entity_type == "product" and entity_name:
            product = self._product(session, table_row, canonical, status)
            session.add(product)
            session.flush()
            self._add_product_aliases(session, product, canonical.get("aliases"))
            return "product"
        if entity_type == "service" and entity_name:
            service = self._service(session, table_row, canonical, status)
            session.add(service)
            session.flush()
            self._add_service_aliases(session, service, canonical.get("aliases"))
            return "service"
        if entity_type == "faq" and canonical.get("question") and canonical.get("answer"):
            category = resolve_category(session, "faq", canonical.get("category"))
            faq = FAQ(
                question=str(canonical["question"]),
                answer=str(canonical["answer"]),
                category=category.display_name or category.source_value,
                category_code=category.code,
                keywords=self._split_values(canonical.get("keywords")),
                is_active=status == "active",
                embedding=embedding,
                source_doc_id=table_row.doc_id,
                source_row_id=table_row.row_id,
                metadata_json={
                    "source_doc_id": str(table_row.doc_id),
                    "source_row_id": str(table_row.row_id),
                },
            )
            session.add(faq)
            session.flush()
            self._add_faq_aliases(session, faq, canonical.get("aliases"))
            return "faq"
        if entity_type == "clinic_info" and canonical.get("key") and canonical.get("value"):
            session.add(
                ClinicInfo(
                    key=str(canonical["key"]),
                    value=str(canonical["value"]),
                    source_doc_id=table_row.doc_id,
                    status=status,
                    metadata_json={
                        "source_row_id": str(table_row.row_id),
                        "source_table_id": str(table_row.table_id),
                    },
                )
            )
            return "clinic_info"
        return None

    def _product(
        self, session: Session, row: TableRow, data: dict[str, Any], status: str
    ) -> Product:
        category = resolve_category(session, "product", data.get("category"))
        return Product(
            name=str(data.get("name")),
            category=category.display_name or category.source_value,
            category_code=category.code,
            source_category=category.source_value,
            brand=self._optional_str(data.get("brand")),
            model=self._optional_str(data.get("model")),
            description=self._optional_str(data.get("description")),
            price=parse_decimal(data.get("price")),
            currency=self._optional_str(data.get("currency")) or "VND",
            quantity=parse_int(data.get("quantity")),
            link=self._optional_str(data.get("link")),
            asset_id=self._asset_id(session, data, row.doc_id),
            image_reference=self._optional_str(data.get("image_reference")),
            source_doc_id=row.doc_id,
            source_row_id=row.row_id,
            status=status,
            metadata_json={"raw_row": data["_raw"]},
        )

    def _service(
        self, session: Session, row: TableRow, data: dict[str, Any], status: str
    ) -> Service:
        symptoms = data.get("symptoms")
        category = resolve_category(session, "service", data.get("category"))
        return Service(
            name=str(data.get("service_name") or data.get("name")),
            category_code=category.code,
            source_category=category.source_value,
            description=self._optional_str(data.get("description")),
            duration_minutes=parse_int(data.get("duration")),
            price=parse_decimal(data.get("price")),
            currency=self._optional_str(data.get("currency")) or "VND",
            symptoms=[item.strip() for item in re.split(r"[,;|]", str(symptoms)) if item.strip()]
            if symptoms
            else None,
            indications=self._split_values(data.get("indications")),
            contraindications=self._split_values(data.get("contraindications")),
            asset_id=self._asset_id(session, data, row.doc_id),
            image_reference=self._optional_str(data.get("image_reference")),
            source_doc_id=row.doc_id,
            source_row_id=row.row_id,
            status=status,
            metadata_json={"raw_row": data["_raw"]},
        )

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return str(value).strip() if value not in (None, "") else None

    @classmethod
    def _asset_id(
        cls,
        session: Session,
        data: dict[str, Any],
        doc_id: uuid.UUID,
    ) -> uuid.UUID | None:
        combined = " ".join(str(value) for value in data["_raw"].values())
        tokens = detect_asset_tokens(combined)
        if tokens:
            return session.scalar(
                select(Asset.asset_id).where(
                    Asset.doc_id == doc_id,
                    Asset.asset_token == tokens[0],
                )
            )
        image_reference = cls._optional_str(data.get("image_reference"))
        if not image_reference:
            return None
        target = image_reference.replace("\\", "/").rsplit("/", 1)[-1].lower()
        assets = session.scalars(select(Asset).where(Asset.doc_id == doc_id)).all()
        for asset in assets:
            metadata = asset.metadata_json or {}
            candidates = {
                str(metadata.get("original_file_name") or "").lower(),
                str(metadata.get("source_ref") or "").replace("\\", "/").rsplit("/", 1)[-1].lower(),
            }
            if target in candidates:
                return asset.asset_id
        return None

    @staticmethod
    def _split_values(value: Any) -> list[str] | None:
        if value in (None, ""):
            return None
        items = [item.strip() for item in re.split(r"[,;|]", str(value)) if item.strip()]
        return items or None

    @classmethod
    def _add_product_aliases(
        cls,
        session: Session,
        product: Product,
        value: Any,
    ) -> None:
        for alias in cls._split_values(value) or []:
            session.add(
                ProductAlias(
                    product_id=product.product_id,
                    alias=alias,
                    normalized_alias=normalize_label(alias),
                )
            )

    @classmethod
    def _add_service_aliases(
        cls,
        session: Session,
        service: Service,
        value: Any,
    ) -> None:
        for alias in cls._split_values(value) or []:
            session.add(
                ServiceAlias(
                    service_id=service.service_id,
                    alias=alias,
                    normalized_alias=normalize_label(alias),
                )
            )

    @classmethod
    def _add_faq_aliases(
        cls,
        session: Session,
        faq: FAQ,
        value: Any,
    ) -> None:
        for variant in cls._split_values(value) or []:
            session.add(
                FAQAlias(
                    faq_id=faq.faq_id,
                    question_variant=variant,
                    normalized_variant=normalize_label(variant),
                )
            )
