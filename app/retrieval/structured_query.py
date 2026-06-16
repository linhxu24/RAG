import re
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CategoryAlias, Product, ProductAlias, ProductCategory
from app.ingestion.table_processor import parse_decimal
from app.retrieval.normalization import normalize_vietnamese


@dataclass(frozen=True)
class ProductQuerySpec:
    category_codes: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    price_min: Decimal | None = None
    price_max: Decimal | None = None
    sort_by: str = "category"
    sort_direction: str = "asc"
    limit: int = 500
    needs_clarification: bool = False
    clarification_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "category_codes": list(self.category_codes),
            "product_ids": list(self.product_ids),
            "price_min": float(self.price_min) if self.price_min is not None else None,
            "price_max": float(self.price_max) if self.price_max is not None else None,
            "sort_by": self.sort_by,
            "sort_direction": self.sort_direction,
            "limit": self.limit,
            "needs_clarification": self.needs_clarification,
            "clarification_message": self.clarification_message,
        }


def parse_product_query(session: Session, query: str) -> ProductQuerySpec:
    normalized = normalize_vietnamese(query)
    category_codes = _category_codes(session, normalized)
    product_ids = _product_ids(session, normalized)
    price_min = _price_bound(normalized, ("tu", "tren", "toi thieu", "it nhat"))
    price_max = _price_bound(normalized, ("duoi", "toi da", "khong qua", "nho hon"))

    sort_requested = any(
        phrase in normalized
        for phrase in ("sap xep", "thu tu", "tang dan", "giam dan", "thap den cao", "cao den thap")
    )
    if any(phrase in normalized for phrase in ("thap den cao", "tang dan", "re nhat")):
        sort_direction = "asc"
    elif any(phrase in normalized for phrase in ("cao den thap", "giam dan", "dat nhat")):
        sort_direction = "desc"
    else:
        sort_direction = "asc"

    if any(word in normalized for word in ("gia", "re", "dat")):
        sort_by = "price"
    elif any(word in normalized for word in ("so luong", "ton kho")):
        sort_by = "quantity"
    elif any(word in normalized for word in ("ten", "chu cai", "alphabet")):
        sort_by = "name"
    else:
        sort_by = "category"

    filter_requested = any(word in normalized for word in ("loc", "chi lay", "thuoc loai"))
    clarification = None
    if sort_requested and not any(
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
    elif filter_requested and not (category_codes or product_ids or price_min or price_max):
        clarification = (
            "Bạn muốn lọc theo danh mục nào, ví dụ bàn chải điện, kem đánh răng "
            "hoặc máy tăm nước?"
        )

    return ProductQuerySpec(
        category_codes=tuple(category_codes),
        product_ids=tuple(product_ids),
        price_min=price_min,
        price_max=price_max,
        sort_by=sort_by,
        sort_direction=sort_direction,
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
