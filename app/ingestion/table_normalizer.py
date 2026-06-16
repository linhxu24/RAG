import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


def normalize_key(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFKD",
        str(value).replace("đ", "d").replace("Đ", "D"),
    )
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")


COLUMN_ALIASES = {
    "name": {"name", "ten", "ten_san_pham", "san_pham", "product", "product_name"},
    "service_name": {"ten_dich_vu", "dich_vu", "service", "service_name"},
    "brand": {"brand", "thuong_hieu", "hang"},
    "model": {"model", "ma_san_pham", "sku"},
    "description": {"description", "mo_ta", "chi_tiet", "noi_dung"},
    "category": {"category", "danh_muc", "loai", "nhom"},
    "price": {"price", "gia", "gia_ban", "chi_phi"},
    "currency": {"currency", "tien_te", "don_vi_tien"},
    "quantity": {"quantity", "so_luong", "ton_kho"},
    "duration": {"duration", "duration_minutes", "thoi_gian", "thoi_luong"},
    "link": {"link", "url", "duong_dan"},
    "symptoms": {"symptoms", "trieu_chung", "chi_dinh"},
    "indications": {"indications", "chi_dinh_dieu_tri", "phu_hop"},
    "contraindications": {"contraindications", "chong_chi_dinh", "khong_phu_hop"},
    "question": {"question", "cau_hoi"},
    "answer": {"answer", "tra_loi", "cau_tra_loi"},
    "keywords": {"keywords", "tu_khoa"},
    "aliases": {"aliases", "alias", "ten_khac", "cau_hoi_tuong_tu"},
    "key": {"key", "thuoc_tinh", "thong_tin"},
    "value": {"value", "gia_tri"},
    "asset": {"asset", "image", "anh", "hinh_anh"},
    "image_reference": {
        "image_reference",
        "image_file",
        "image_filename",
        "ten_file_anh",
        "file_anh",
    },
}

CLINIC_KEYS = {
    "ten_nha_khoa",
    "ten_phong_kham",
    "so_dien_thoai",
    "dien_thoai",
    "phone",
    "email",
    "dia_chi",
    "address",
    "thoi_gian_lam_viec",
    "gio_lam_viec",
    "opening_hours",
    "facebook",
    "zalo",
    "website",
}


@dataclass(frozen=True)
class TableClassification:
    entity_type: str | None
    confidence: float
    reasons: list[str] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)
    requires_review: bool = False

    def as_metadata(self) -> dict[str, Any]:
        return {
            "detected_entity_type": self.entity_type,
            "classification_confidence": self.confidence,
            "classification_reasons": self.reasons,
            "column_mapping": self.column_mapping,
            "requires_review": self.requires_review,
        }


@dataclass(frozen=True)
class NormalizedTable:
    rows: list[dict[str, Any]]
    classification: TableClassification
    warnings: list[str] = field(default_factory=list)


def canonicalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {normalize_key(key): value for key, value in row.items()}
    result: dict[str, Any] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized and normalized[alias] not in (None, ""):
                result[canonical] = normalized[alias]
                break
    result["_raw"] = row
    return result


def normalize_table(
    rows: list[dict[str, Any]],
    table_name: str | None = None,
    document_type: str = "auto",
) -> NormalizedTable:
    clean_rows = [_json_safe_row(row) for row in rows if _has_values(row)]
    if not clean_rows:
        return NormalizedTable(
            rows=[],
            classification=TableClassification(
                entity_type=None,
                confidence=0.0,
                reasons=["table_has_no_non_empty_rows"],
                requires_review=True,
            ),
            warnings=["Table has no non-empty rows"],
        )

    key_value = _normalize_key_value_layout(clean_rows)
    if key_value is not None:
        return key_value

    schedule = _normalize_schedule_layout(clean_rows)
    if schedule is not None:
        return schedule

    classification = classify_table(clean_rows, table_name, document_type)
    return NormalizedTable(rows=clean_rows, classification=classification)


def classify_table(
    rows: list[dict[str, Any]],
    table_name: str | None = None,
    document_type: str = "auto",
) -> TableClassification:
    keys = {normalize_key(key) for row in rows for key in row}
    name = normalize_key(table_name or "")
    mapping = _column_mapping(rows)

    if document_type in {"product", "service", "faq", "clinic_info"}:
        return TableClassification(
            entity_type=document_type,
            confidence=0.99,
            reasons=["explicit_document_type_override"],
            column_mapping=mapping,
        )

    if keys & COLUMN_ALIASES["question"] and keys & COLUMN_ALIASES["answer"]:
        return TableClassification(
            entity_type="faq",
            confidence=0.98,
            reasons=["question_and_answer_columns"],
            column_mapping=mapping,
        )

    has_generic_name = bool(keys & COLUMN_ALIASES["name"])
    has_service_name = bool(keys & COLUMN_ALIASES["service_name"])
    service_specific = keys & (
        COLUMN_ALIASES["duration"]
        | COLUMN_ALIASES["symptoms"]
        | COLUMN_ALIASES["indications"]
        | COLUMN_ALIASES["contraindications"]
    )
    product_specific = keys & (
        COLUMN_ALIASES["brand"]
        | COLUMN_ALIASES["model"]
        | COLUMN_ALIASES["quantity"]
        | COLUMN_ALIASES["link"]
    )

    if has_service_name or (has_generic_name and service_specific):
        return TableClassification(
            entity_type="service",
            confidence=0.97 if has_service_name else 0.96,
            reasons=[
                "service_name_column"
                if has_service_name
                else "generic_name_and_service_attribute_columns"
            ],
            column_mapping=mapping,
        )

    if "service" in name or "dich_vu" in name:
        return TableClassification(
            entity_type="service",
            confidence=0.82,
            reasons=["service_table_name"],
            column_mapping=mapping,
            requires_review=True,
        )

    if has_generic_name and product_specific:
        return TableClassification(
            entity_type="product",
            confidence=0.96,
            reasons=["generic_name_and_product_attribute_columns"],
            column_mapping=mapping,
        )

    if "product" in name or "san_pham" in name:
        return TableClassification(
            entity_type="product",
            confidence=0.82,
            reasons=["product_table_name"],
            column_mapping=mapping,
            requires_review=True,
        )

    if keys & COLUMN_ALIASES["key"] and keys & COLUMN_ALIASES["value"]:
        return TableClassification(
            entity_type="clinic_info",
            confidence=0.96,
            reasons=["key_value_columns"],
            column_mapping=mapping,
        )

    return TableClassification(
        entity_type=None,
        confidence=0.0,
        reasons=["no_supported_business_schema_detected"],
        column_mapping=mapping,
        requires_review=True,
    )


def _normalize_key_value_layout(rows: list[dict[str, Any]]) -> NormalizedTable | None:
    columns = list(rows[0])
    if len(columns) != 2 or any(list(row) != columns for row in rows):
        return None
    normalized_columns = [normalize_key(column) for column in columns]
    positional = all(
        not column or column.isdigit() or column.startswith("unnamed")
        for column in normalized_columns
    )
    if not positional:
        return None

    first_values = [normalize_key(row.get(columns[0], "")) for row in rows]
    clinic_matches = sum(value in CLINIC_KEYS for value in first_values)
    if clinic_matches < max(2, len(rows) // 3):
        return None

    normalized_rows = [
        {"key": row.get(columns[0], ""), "value": row.get(columns[1], "")} for row in rows
    ]
    return NormalizedTable(
        rows=normalized_rows,
        classification=TableClassification(
            entity_type="clinic_info",
            confidence=0.97,
            reasons=["positional_two_column_clinic_key_value_layout"],
            column_mapping={columns[0]: "key", columns[1]: "value"},
        ),
    )


def _normalize_schedule_layout(rows: list[dict[str, Any]]) -> NormalizedTable | None:
    mapping = _column_mapping(rows)
    normalized_headers = {normalize_key(key): key for key in rows[0]}
    slot_header = next(
        (
            normalized_headers[key]
            for key in ("khung_gio", "buoi", "time_slot", "shift")
            if key in normalized_headers
        ),
        None,
    )
    value_header = next(
        (
            normalized_headers[key]
            for key in ("mo_ta", "thoi_gian", "gio", "hours", "value")
            if key in normalized_headers
        ),
        None,
    )
    if slot_header is None or value_header is None:
        return None

    normalized_rows = [
        {
            "key": f"Giờ làm việc - {row.get(slot_header, '')}".strip(" -"),
            "value": row.get(value_header, ""),
        }
        for row in rows
    ]
    mapping.update({slot_header: "key", value_header: "value"})
    return NormalizedTable(
        rows=normalized_rows,
        classification=TableClassification(
            entity_type="clinic_info",
            confidence=0.93,
            reasons=["opening_hours_schedule_layout"],
            column_mapping=mapping,
        ),
    )


def _column_mapping(rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_key in rows[0]:
        normalized = normalize_key(raw_key)
        for canonical, aliases in COLUMN_ALIASES.items():
            if normalized in aliases:
                mapping[str(raw_key)] = canonical
                break
    return mapping


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        safe[str(key)] = value
    return safe


def _has_values(row: dict[str, Any]) -> bool:
    return any(value not in (None, "") for value in row.values())
