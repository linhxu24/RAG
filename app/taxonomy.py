import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    CategoryAlias,
    FAQCategory,
    ProductCategory,
    ServiceCategory,
)


def normalize_label(value: str | None) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or "").lower())
    ascii_value = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", ascii_value)).strip()


@dataclass(frozen=True)
class CategoryResolution:
    code: str | None
    display_name: str | None
    source_value: str | None
    matched_by: str | None = None


def resolve_category(
    session: Session,
    entity_type: str,
    value: str | None,
) -> CategoryResolution:
    source = str(value).strip() if value not in (None, "") else None
    if source is None:
        return CategoryResolution(None, None, None)
    normalized = normalize_label(source)
    model = {
        "product": ProductCategory,
        "service": ServiceCategory,
        "faq": FAQCategory,
    }.get(entity_type)
    if model is None:
        return CategoryResolution(None, source, source)

    direct = session.scalar(
        select(model).where(
            model.status == "active",
            (model.code == source.upper()) | (model.display_name.ilike(source)),
        )
    )
    if direct is not None:
        return CategoryResolution(direct.code, direct.display_name, source, "direct")

    alias = session.scalar(
        select(CategoryAlias).where(
            CategoryAlias.entity_type == entity_type,
            CategoryAlias.normalized_alias == normalized,
        )
    )
    if alias is None:
        return CategoryResolution(None, source, source)
    category = session.get(model, alias.category_code)
    if category is None or category.status != "active":
        return CategoryResolution(None, source, source)
    return CategoryResolution(category.code, category.display_name, source, "alias")
