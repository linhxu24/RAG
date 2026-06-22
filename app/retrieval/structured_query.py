import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    CategoryAlias,
    Product,
    ProductAlias,
    ProductCategory,
    Service,
    ServiceCategory,
)
from app.ingestion.table_processor import parse_decimal
from app.retrieval.normalization import normalize_vietnamese


@dataclass(frozen=True)
class ProductQuerySpec:
    category_codes: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    brand_terms: tuple[str, ...] = ()
    feature_terms: tuple[str, ...] = ()
    price_min: Decimal | None = None
    price_max: Decimal | None = None
    quantity_min: int | None = None
    quantity_max: int | None = None
    sort_by: str = "category"
    sort_direction: str = "asc"
    limit: int = 500
    needs_clarification: bool = False
    clarification_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "category_codes": list(self.category_codes),
            "product_ids": list(self.product_ids),
            "brand_terms": list(self.brand_terms),
            "feature_terms": list(self.feature_terms),
            "price_min": float(self.price_min) if self.price_min is not None else None,
            "price_max": float(self.price_max) if self.price_max is not None else None,
            "quantity_min": self.quantity_min,
            "quantity_max": self.quantity_max,
            "sort_by": self.sort_by,
            "sort_direction": self.sort_direction,
            "limit": self.limit,
            "needs_clarification": self.needs_clarification,
            "clarification_message": self.clarification_message,
        }


@dataclass(frozen=True)
class ServiceQuerySpec:
    category_codes: tuple[str, ...] = ()
    category_terms: tuple[str, ...] = ()
    service_ids: tuple[str, ...] = ()
    feature_terms: tuple[str, ...] = ()
    symptom_terms: tuple[str, ...] = ()
    price_min: Decimal | None = None
    price_max: Decimal | None = None
    duration_min: int | None = None
    duration_max: int | None = None
    sort_by: str = "name"
    sort_direction: str = "asc"
    limit: int = 500
    needs_clarification: bool = False
    clarification_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "category_codes": list(self.category_codes),
            "category_terms": list(self.category_terms),
            "service_ids": list(self.service_ids),
            "feature_terms": list(self.feature_terms),
            "symptom_terms": list(self.symptom_terms),
            "price_min": float(self.price_min) if self.price_min is not None else None,
            "price_max": float(self.price_max) if self.price_max is not None else None,
            "duration_min": self.duration_min,
            "duration_max": self.duration_max,
            "sort_by": self.sort_by,
            "sort_direction": self.sort_direction,
            "limit": self.limit,
            "needs_clarification": self.needs_clarification,
            "clarification_message": self.clarification_message,
        }


def parse_product_query(
    session: Session,
    query: str,
    constraints: dict[str, Any] | None = None,
    sort: dict[str, Any] | None = None,
    limit: int | None = None,
) -> ProductQuerySpec:
    normalized = normalize_vietnamese(query)
    constraints = constraints or {}
    category_terms = _constraint_strings(
        constraints,
        "category_terms",
        "categories",
        "product_categories",
        "category",
        "loai_san_pham",
    )
    category_codes = _dedupe(
        [
            *_category_codes(session, normalized),
            *_constraint_strings(constraints, "category_codes"),
            *_category_codes(session, normalize_vietnamese(" ".join(category_terms))),
        ]
    )
    product_name_terms = _constraint_strings(
        constraints,
        "product_names",
        "name_terms",
        "names",
        "entities",
    )
    product_ids = _dedupe(
        [
            *_product_ids(session, normalized),
            *_constraint_strings(constraints, "product_ids"),
            *_product_ids(session, normalize_vietnamese(" ".join(product_name_terms))),
        ]
    )
    brand_terms = tuple(
        _dedupe(_constraint_strings(constraints, "brands", "brand_terms", "brand"))
    )
    feature_terms = tuple(
        _dedupe(
            [
                *_constraint_strings(
                    constraints,
                    "feature_terms",
                    "features",
                    "use_cases",
                    "benefits",
                    "cong_dung",
                    "tinh_nang",
                ),
                *_product_feature_terms_from_query(normalized),
            ]
        )
    )
    price_min = _coalesce_decimal(
        _constraint_value(constraints, "price_min", "min_price"),
        _nested_constraint_value(constraints, "price", "min"),
        _price_bound(normalized, ("tu", "tren", "toi thieu", "it nhat")),
    )
    price_max = _coalesce_decimal(
        _constraint_value(constraints, "price_max", "max_price"),
        _nested_constraint_value(constraints, "price", "max"),
        _price_bound(normalized, ("duoi", "toi da", "khong qua", "nho hon")),
    )
    quantity_min = _coalesce_int(
        _constraint_value(constraints, "quantity_min", "min_quantity"),
        _nested_constraint_value(constraints, "quantity", "min"),
        1 if _stock_available_requested(normalized, constraints) else None,
    )
    quantity_max = _coalesce_int(
        _constraint_value(constraints, "quantity_max", "max_quantity"),
        _nested_constraint_value(constraints, "quantity", "max"),
    )

    sort_requested = _sort_requested(normalized, sort)
    sort_field = _sort_field(sort)
    if _sort_direction(sort) == "asc" or any(
        phrase in normalized for phrase in ("thap den cao", "tang dan", "re nhat")
    ):
        sort_direction = "asc"
    elif _sort_direction(sort) == "desc" or any(
        phrase in normalized for phrase in ("cao den thap", "giam dan", "dat nhat")
    ):
        sort_direction = "desc"
    else:
        sort_direction = "asc"

    if sort_field in {"price", "quantity", "name", "category"}:
        sort_by = sort_field
    elif any(word in normalized for word in ("gia", "re", "dat")):
        sort_by = "price"
    elif any(word in normalized for word in ("so luong", "ton kho")):
        sort_by = "quantity"
    elif any(word in normalized for word in ("ten", "chu cai", "alphabet")):
        sort_by = "name"
    else:
        sort_by = "category"

    filter_requested = any(word in normalized for word in ("loc", "chi lay", "thuoc loai"))
    clarification = None
    if sort_requested and _sort_direction(sort) is None and not any(
        phrase in normalized
        for phrase in (
            "tang dan",
            "giam dan",
            "thap den cao",
            "cao den thap",
            "re nhat",
            "dat nhat",
        )
    ):
        clarification = (
            "Bạn muốn sắp xếp tăng dần hay giảm dần? "
            "Ví dụ: giá từ thấp đến cao hoặc giá từ cao đến thấp."
        )
    elif filter_requested and not (
        category_codes
        or product_ids
        or brand_terms
        or feature_terms
        or price_min
        or price_max
        or quantity_min
        or quantity_max
    ):
        clarification = (
            "Bạn muốn lọc theo danh mục nào, ví dụ bàn chải điện, kem đánh răng "
            "hoặc máy tăm nước?"
        )

    return ProductQuerySpec(
        category_codes=tuple(category_codes),
        product_ids=tuple(product_ids),
        brand_terms=brand_terms,
        feature_terms=feature_terms,
        price_min=price_min,
        price_max=price_max,
        quantity_min=quantity_min,
        quantity_max=quantity_max,
        sort_by=sort_by,
        sort_direction=sort_direction,
        limit=limit or _constraint_limit(constraints) or 500,
        needs_clarification=clarification is not None,
        clarification_message=clarification,
    )


def parse_service_query(
    session: Session,
    query: str,
    constraints: dict[str, Any] | None = None,
    sort: dict[str, Any] | None = None,
    limit: int | None = None,
) -> ServiceQuerySpec:
    normalized = normalize_vietnamese(query)
    padded = f" {normalized} "
    constraints = constraints or {}
    raw_category_terms = _constraint_strings(
        constraints,
        "category_terms",
        "categories",
        "service_categories",
        "category",
        "loai_dich_vu",
    )
    parsed_category_codes, parsed_category_terms = _service_category_match(session, normalized)
    constraint_category_codes, constraint_category_terms = _service_category_match(
        session,
        normalize_vietnamese(" ".join(raw_category_terms)),
    )
    category_codes = _dedupe(
        [
            *parsed_category_codes,
            *constraint_category_codes,
            *_constraint_strings(constraints, "category_codes"),
        ]
    )
    category_terms = _dedupe([*parsed_category_terms, *constraint_category_terms])
    service_name_terms = _constraint_strings(
        constraints,
        "service_names",
        "name_terms",
        "names",
        "entities",
    )
    service_ids = _dedupe(
        [
            *_service_ids(session, normalized),
            *_constraint_strings(constraints, "service_ids"),
            *_service_ids(session, normalize_vietnamese(" ".join(service_name_terms))),
        ]
    )
    feature_terms = tuple(
        _dedupe(
            [
                *_constraint_strings(
                    constraints,
                    "feature_terms",
                    "features",
                    "goals",
                    "benefits",
                    "muc_tieu",
                ),
                *_service_feature_terms_from_query(normalized),
            ]
        )
    )
    symptom_terms = tuple(
        _dedupe(
            _constraint_strings(
                constraints,
                "symptom_terms",
                "symptoms",
                "indications",
                "trieu_chung",
            )
        )
    )
    price_min = _coalesce_decimal(
        _constraint_value(constraints, "price_min", "min_price"),
        _nested_constraint_value(constraints, "price", "min"),
        _price_bound(normalized, ("tu", "tren", "toi thieu", "it nhat")),
    )
    price_max = _coalesce_decimal(
        _constraint_value(constraints, "price_max", "max_price"),
        _nested_constraint_value(constraints, "price", "max"),
        _price_bound(normalized, ("duoi", "toi da", "khong qua", "nho hon")),
    )
    has_duration_context = _has_duration_context(normalized)
    duration_min = _coalesce_int(
        _constraint_value(constraints, "duration_min", "min_duration"),
        _nested_constraint_value(constraints, "duration", "min"),
        _duration_bound(normalized, ("tu", "tren", "toi thieu", "it nhat"))
        if has_duration_context
        else None,
    )
    duration_max = _coalesce_int(
        _constraint_value(constraints, "duration_max", "max_duration"),
        _nested_constraint_value(constraints, "duration", "max"),
        _duration_bound(normalized, ("duoi", "toi da", "khong qua", "nho hon"))
        if has_duration_context
        else None,
    )

    sort_requested = _sort_requested(normalized, sort)
    sort_field = _sort_field(sort)
    if _sort_direction(sort) == "desc" or any(
        phrase in normalized for phrase in ("cao den thap", "giam dan", "dat nhat", "lau nhat")
    ):
        sort_direction = "desc"
    else:
        sort_direction = "asc"

    if sort_field in {"price", "duration", "name", "category"}:
        sort_by = sort_field
    elif " gia " in padded or any(word in normalized for word in ("chi phi", "re", "dat")):
        sort_by = "price"
    elif any(word in normalized for word in ("thoi gian", "mat bao lau", "nhanh", "lau")):
        sort_by = "duration"
    elif any(word in normalized for word in ("danh muc", "loai", "nhom")):
        sort_by = "category"
    else:
        sort_by = "name"

    filter_requested = any(
        phrase in normalized
        for phrase in (
            "loc",
            "chi lay",
            "thuoc loai",
            "lien quan",
            "phu hop",
            "dich vu nao",
        )
    )
    clarification = None
    if sort_requested and _sort_direction(sort) is None and not any(
        phrase in normalized
        for phrase in (
            "tang dan",
            "giam dan",
            "thap den cao",
            "cao den thap",
            "ngan nhat",
            "lau nhat",
            "re nhat",
            "dat nhat",
        )
    ):
        clarification = (
            "Bạn muốn sắp xếp tăng dần hay giảm dần? Ví dụ: giá từ thấp đến cao "
            "hoặc thời lượng ngắn nhất trước."
        )
    elif filter_requested and not (
        category_codes
        or service_ids
        or feature_terms
        or symptom_terms
        or price_min
        or price_max
        or duration_min
        or duration_max
    ):
        clarification = (
            "Bạn muốn lọc dịch vụ theo nhóm nào, ví dụ implant, tẩy trắng, niềng răng "
            "hoặc nhổ răng?"
        )

    return ServiceQuerySpec(
        category_codes=tuple(category_codes),
        category_terms=tuple(category_terms),
        service_ids=tuple(service_ids),
        feature_terms=feature_terms,
        symptom_terms=symptom_terms,
        price_min=price_min,
        price_max=price_max,
        duration_min=duration_min,
        duration_max=duration_max,
        sort_by=sort_by,
        sort_direction=sort_direction,
        limit=limit or _constraint_limit(constraints) or 500,
        needs_clarification=clarification is not None,
        clarification_message=clarification,
    )


def active_product_category_terms(session: Session) -> list[str]:
    aliases = session.scalars(
        select(CategoryAlias.alias).where(CategoryAlias.entity_type == "product")
    ).all()
    names = session.scalars(
        select(ProductCategory.display_name).where(ProductCategory.status == "active")
    ).all()
    return list(dict.fromkeys([*names, *aliases]))


def active_service_category_terms(session: Session) -> list[str]:
    aliases = session.scalars(
        select(CategoryAlias.alias).where(CategoryAlias.entity_type == "service")
    ).all()
    names = session.scalars(
        select(ServiceCategory.display_name).where(ServiceCategory.status == "active")
    ).all()
    return list(dict.fromkeys([*names, *aliases]))


def _category_codes(session: Session, normalized_query: str) -> list[str]:
    aliases = session.scalars(
        select(CategoryAlias).where(CategoryAlias.entity_type == "product")
    ).all()
    matches = [
        alias
        for alias in aliases
        if alias.normalized_alias and alias.normalized_alias in normalized_query
    ]
    if not matches:
        return []
    longest = max(len(alias.normalized_alias) for alias in matches)
    selected = {
        alias.category_code
        for alias in matches
        if len(alias.normalized_alias) >= longest - 2
    }
    descendants = session.scalars(
        select(ProductCategory.code).where(ProductCategory.parent_code.in_(selected))
    ).all()
    return sorted(selected | set(descendants))


def _service_category_match(
    session: Session,
    normalized_query: str,
) -> tuple[list[str], list[str]]:
    aliases = session.scalars(
        select(CategoryAlias).where(CategoryAlias.entity_type == "service")
    ).all()
    matches = [
        alias
        for alias in aliases
        if alias.normalized_alias and alias.normalized_alias in normalized_query
    ]
    if not matches:
        return [], []
    longest = max(len(alias.normalized_alias) for alias in matches)
    selected = {
        alias.category_code
        for alias in matches
        if len(alias.normalized_alias) >= longest - 2
    }
    descendants = session.scalars(
        select(ServiceCategory.code).where(ServiceCategory.parent_code.in_(selected))
    ).all()
    category_codes = sorted(selected | set(descendants))
    terms = sorted(
        {
            alias.normalized_alias
            for alias in matches
            if alias.category_code in selected and alias.normalized_alias
        }
        | {normalize_vietnamese(code) for code in category_codes}
    )
    return category_codes, terms


def _product_ids(session: Session, normalized_query: str) -> list[str]:
    records = session.scalars(select(Product).where(Product.status == "active")).all()
    aliases = session.scalars(select(ProductAlias)).all()
    alias_by_product: dict[object, list[str]] = {}
    for alias in aliases:
        alias_by_product.setdefault(alias.product_id, []).append(alias.normalized_alias)
    return [
        str(product.product_id)
        for product in records
        if normalize_vietnamese(product.name) in normalized_query
        or any(
            alias and alias in normalized_query
            for alias in alias_by_product.get(product.product_id, [])
        )
    ]


def _service_ids(session: Session, normalized_query: str) -> list[str]:
    records = session.scalars(select(Service).where(Service.status == "active")).all()
    return [
        str(service.service_id)
        for service in records
        if normalize_vietnamese(service.name) in normalized_query
    ]


def _price_bound(normalized_query: str, prefixes: tuple[str, ...]) -> Decimal | None:
    prefix_pattern = "|".join(re.escape(prefix) for prefix in prefixes)
    match = re.search(
        rf"(?:{prefix_pattern})\s+(\d[\d.,]*)\s*(trieu|tram nghin|nghin|k)?",
        normalized_query,
    )
    if match is None:
        return None
    value = parse_decimal(match.group(1))
    if value is None:
        return None
    unit = match.group(2)
    if unit == "trieu":
        return value * 1_000_000
    if unit == "tram nghin":
        return value * 100_000
    if unit in {"nghin", "k"}:
        return value * 1_000
    return value


def _duration_bound(normalized_query: str, prefixes: tuple[str, ...]) -> int | None:
    prefix_pattern = "|".join(re.escape(prefix) for prefix in prefixes)
    match = re.search(
        rf"(?:{prefix_pattern})\s+(\d[\d.,]*)\s*(gio|tieng|phut)?",
        normalized_query,
    )
    if match is None:
        return None
    value = parse_decimal(match.group(1))
    if value is None:
        return None
    unit = match.group(2)
    minutes = value * 60 if unit in {"gio", "tieng"} else value
    return int(minutes)


def _constraint_strings(constraints: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(_string_values(_constraint_value(constraints, key)))
    return _dedupe(values)


def _constraint_value(constraints: dict[str, Any], *keys: str):
    for key in keys:
        if key in constraints:
            return constraints[key]
    return None


def _nested_constraint_value(constraints: dict[str, Any], key: str, child_key: str):
    value = constraints.get(key)
    if isinstance(value, dict):
        return value.get(child_key)
    return None


def _string_values(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_string_values(item))
        return values
    text = str(value).strip()
    return [text] if text else []


def _coalesce_decimal(*values: object) -> Decimal | None:
    for value in values:
        parsed = _decimal_value(value)
        if parsed is not None:
            return parsed
    return None


def _decimal_value(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))
    if isinstance(value, str):
        normalized = normalize_vietnamese(value)
        bound = _price_bound(f"duoi {normalized}", ("duoi",))
        if bound is not None:
            return bound
        return parse_decimal(value)
    return None


def _coalesce_int(*values: object) -> int | None:
    for value in values:
        parsed = _int_value(value)
        if parsed is not None:
            return parsed
    return None


def _int_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, str):
        parsed = parse_decimal(value)
        return int(parsed) if parsed is not None else None
    return None


def _constraint_limit(constraints: dict[str, Any]) -> int | None:
    value = _constraint_value(constraints, "limit", "top_k", "top")
    parsed = _int_value(value)
    if parsed is None:
        return None
    return max(1, min(parsed, 500))


def _sort_field(sort: dict[str, Any] | None) -> str | None:
    if not isinstance(sort, dict):
        return None
    value = sort.get("field") or sort.get("sort_by")
    if not value:
        return None
    normalized = normalize_vietnamese(str(value))
    mapping = {
        "gia": "price",
        "price": "price",
        "chi phi": "price",
        "so luong": "quantity",
        "ton kho": "quantity",
        "quantity": "quantity",
        "thoi gian": "duration",
        "duration": "duration",
        "ten": "name",
        "name": "name",
        "danh muc": "category",
        "category": "category",
    }
    return mapping.get(normalized, normalized)


def _sort_direction(sort: dict[str, Any] | None) -> str | None:
    if not isinstance(sort, dict):
        return None
    value = sort.get("direction") or sort.get("sort_direction")
    if not value:
        return None
    normalized = normalize_vietnamese(str(value))
    if normalized in {"asc", "ascending", "tang", "tang dan", "thap den cao"}:
        return "asc"
    if normalized in {"desc", "descending", "giam", "giam dan", "cao den thap"}:
        return "desc"
    return None


def _sort_requested(normalized_query: str, sort: dict[str, Any] | None) -> bool:
    if isinstance(sort, dict) and any(
        sort.get(key)
        for key in ("field", "sort_by", "direction", "sort_direction")
    ):
        return True
    return any(
        phrase in normalized_query
        for phrase in (
            "sap xep",
            "thu tu",
            "tang dan",
            "giam dan",
            "thap den cao",
            "cao den thap",
            "ngan nhat",
            "lau nhat",
        )
    )


def _stock_available_requested(
    normalized_query: str,
    constraints: dict[str, Any],
) -> bool:
    stock = str(_constraint_value(constraints, "stock", "availability") or "").lower()
    if stock in {"available", "in_stock", "con_hang", "còn hàng"}:
        return True
    return any(phrase in normalized_query for phrase in ("con hang", "ton kho", "san co"))


def _product_feature_terms_from_query(normalized_query: str) -> list[str]:
    phrases = (
        "lam trang",
        "rang nhay cam",
        "e buot",
        "nieng rang",
        "ve sinh ke rang",
        "ke rang",
        "nuoc suc mieng",
        "hoi mieng",
    )
    return [phrase for phrase in phrases if phrase in normalized_query]


def _service_feature_terms_from_query(normalized_query: str) -> list[str]:
    phrases = (
        "implant",
        "trong rang",
        "tay trang",
        "lam trang",
        "nieng rang",
        "chinh nha",
        "nho rang",
        "noi nha",
        "dieu tri tuy",
        "ve sinh rang",
        "cao voi",
        "phuc hoi",
    )
    return [phrase for phrase in phrases if phrase in normalized_query]


def _has_duration_context(normalized_query: str) -> bool:
    return any(
        phrase in normalized_query
        for phrase in (
            "thoi gian",
            "mat bao lau",
            "bao lau",
            "keo dai",
            "phut",
            "gio",
            "tieng",
            "nhanh",
            "lau nhat",
            "ngan nhat",
        )
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
