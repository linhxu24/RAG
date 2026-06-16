from decimal import Decimal
from typing import Any

from rapidfuzz import fuzz
from sqlalchemy import asc, desc, select
from sqlalchemy.orm import Session

from app.constants import Intent
from app.db.models import FAQ, Asset, ClinicInfo, FAQAlias, Product, Service
from app.retrieval.normalization import (
    normalize_vietnamese,
    search_query_tokens,
)
from app.retrieval.structured_query import ProductQuerySpec, parse_product_query
from app.retrieval.types import RetrievalResult


def serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


class StructuredRetriever:
    def active_names(self, session: Session) -> tuple[list[str], list[str]]:
        products = list(
            session.scalars(
                select(Product.name)
                .where(Product.status == "active")
                .distinct()
            ).all()
        )
        services = list(
            session.scalars(
                select(Service.name)
                .where(Service.status == "active")
                .distinct()
            ).all()
        )
        return products, services

    def retrieve(
        self,
        session: Session,
        intent: Intent,
        query: str,
        entities: list[str],
    ) -> list[RetrievalResult]:
        if intent == Intent.PRODUCT_LIST:
            specification = parse_product_query(session, query)
            if specification.needs_clarification:
                return []
            return [
                self._product_result(session, item, 1.0)
                for item in self.list_products(session, specification)
            ]
        if intent == Intent.SERVICE_LIST:
            return [
                self._service_result(session, item, 1.0) for item in self.list_services(session)
            ]
        if intent == Intent.PRODUCT_DETAIL:
            product = self.get_product(session, entities[0] if entities else query)
            return [self._product_result(session, product[0], product[1])] if product else []
        if intent == Intent.PRODUCT_COMPARE:
            results = []
            for entity in entities:
                product = self.get_product(session, entity)
                if product:
                    results.append(self._product_result(session, product[0], product[1]))
            return results
        if intent == Intent.SERVICE_DETAIL:
            service = self.get_service(session, entities[0] if entities else query)
            return [self._service_result(session, service[0], service[1])] if service else []
        if intent == Intent.CLINIC_INFO:
            return self.clinic_info(session, query)
        if intent == Intent.FAQ:
            faq = self.get_faq(session, query)
            return [self._faq_result(faq[0], faq[1])] if faq else []
        return []

    def list_products(
        self,
        session: Session,
        specification: ProductQuerySpec | None = None,
    ) -> list[Product]:
        specification = specification or ProductQuerySpec()
        statement = select(Product).where(Product.status == "active")
        if specification.category_codes:
            statement = statement.where(
                Product.category_code.in_(specification.category_codes)
            )
        if specification.product_ids:
            statement = statement.where(
                Product.product_id.in_(specification.product_ids)
            )
        if specification.price_min is not None:
            statement = statement.where(Product.price >= specification.price_min)
        if specification.price_max is not None:
            statement = statement.where(Product.price <= specification.price_max)
        sort_columns = {
            "price": Product.price,
            "quantity": Product.quantity,
            "name": Product.name,
            "category": Product.category,
        }
        sort_column = sort_columns.get(specification.sort_by, Product.category)
        ordering = desc(sort_column) if specification.sort_direction == "desc" else asc(sort_column)
        return list(
            session.scalars(
                statement.order_by(ordering.nulls_last(), Product.name)
                .limit(specification.limit)
            ).all()
        )

    def list_services(self, session: Session) -> list[Service]:
        return list(
            session.scalars(
                select(Service).where(Service.status == "active").order_by(Service.name)
            ).all()
        )

    def get_product(self, session: Session, name_or_query: str) -> tuple[Product, float] | None:
        products = self.list_products(session)
        return self._best_match(products, name_or_query)

    def get_service(self, session: Session, name_or_query: str) -> tuple[Service, float] | None:
        services = self.list_services(session)
        return self._best_match(services, name_or_query)

    def get_faq(self, session: Session, query: str) -> tuple[FAQ, float] | None:
        faqs = list(session.scalars(select(FAQ).where(FAQ.is_active.is_(True))).all())
        if not faqs:
            return None
        aliases = session.scalars(select(FAQAlias)).all()
        alias_by_faq: dict[object, list[str]] = {}
        for alias in aliases:
            alias_by_faq.setdefault(alias.faq_id, []).append(alias.question_variant)
        scored = []
        for faq in faqs:
            candidates = [
                faq.question,
                *(alias_by_faq.get(faq.faq_id, [])),
                *(faq.keywords or []),
            ]
            scored.append(
                (faq, max(self._faq_match_score(query, candidate) for candidate in candidates))
            )
        best = max(scored, key=lambda item: item[1])
        return best if best[1] >= 0.72 else None

    @staticmethod
    def _faq_match_score(query: str, question: str) -> float:
        fuzzy_score = fuzz.WRatio(query.lower(), question.lower()) / 100.0
        query_tokens = set(search_query_tokens(query))
        question_tokens = set(search_query_tokens(question))
        overlap = (
            len(query_tokens & question_tokens)
            / max(1, min(len(query_tokens), len(question_tokens)))
        )
        return 0.8 * fuzzy_score + 0.2 * overlap

    def clinic_info(self, session: Session, query: str) -> list[RetrievalResult]:
        records = list(
            session.scalars(
                select(ClinicInfo).where(ClinicInfo.status == "active").order_by(ClinicInfo.key)
            ).all()
        )
        relevant = [
            record
            for record in records
            if record.key.lower() in query.lower()
            or any(term in query.lower() for term in self._clinic_synonyms(record.key))
        ]
        selected = relevant or records
        return [
            RetrievalResult(
                source_type="clinic_info",
                source_id=str(item.id),
                text=f"{item.key}: {item.value}",
                score=1.0 if item in relevant else 0.8,
                raw_json={"key": item.key, "value": item.value},
                source={"doc_id": str(item.source_doc_id) if item.source_doc_id else None},
                canonical_key=f"clinic_info:{item.id}",
            )
            for item in selected
        ]

    @staticmethod
    def _clinic_synonyms(key: str) -> tuple[str, ...]:
        normalized = key.lower()
        if any(word in normalized for word in ("phone", "hotline", "điện thoại")):
            return ("số điện thoại", "hotline", "liên hệ")
        if any(word in normalized for word in ("address", "địa chỉ")):
            return ("địa chỉ", "ở đâu")
        if any(word in normalized for word in ("hours", "giờ")):
            return ("giờ làm", "mở cửa")
        return (normalized,)

    @staticmethod
    def _best_match(records: list[Any], query: str) -> tuple[Any, float] | None:
        if not records:
            return None
        normalized = normalize_vietnamese(query)
        exact = [
            record
            for record in records
            if normalize_vietnamese(record.name) == normalized
        ]
        if exact:
            return exact[0], 1.0
        contained = [
            record
            for record in records
            if normalize_vietnamese(record.name) in normalized
        ]
        if contained:
            return max(contained, key=lambda item: len(item.name)), 0.96
        scored = [
            (
                record,
                fuzz.WRatio(normalized, normalize_vietnamese(record.name)) / 100.0,
            )
            for record in records
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best = scored[0]
        if best[1] < 0.72:
            return None
        if len(scored) > 1 and best[1] - scored[1][1] < 0.08:
            return None
        return best

    def _product_result(self, session: Session, product: Product, score: float) -> RetrievalResult:
        asset_token = self._asset_token(session, product.asset_id)
        fields = [
            f"Sản phẩm: {product.name}",
            f"Danh mục: {product.category}" if product.category else None,
            f"Thương hiệu: {product.brand}" if product.brand else None,
            f"Model: {product.model}" if product.model else None,
            f"Mô tả: {product.description}" if product.description else None,
            f"Giá: {product.price} {product.currency}"
            if product.price is not None
            else None,
            f"Số lượng: {product.quantity}" if product.quantity is not None else None,
            f"Link: {product.link}" if product.link else None,
            f"Ảnh: {asset_token}" if asset_token else None,
        ]
        return RetrievalResult(
            source_type="product",
            source_id=str(product.product_id),
            text=". ".join(field for field in fields if field),
            score=score,
            raw_json={
                "name": product.name,
                "category": product.category,
                "category_code": product.category_code,
                "brand": product.brand,
                "model": product.model,
                "description": product.description,
                "price": serialize_value(product.price),
                "currency": product.currency,
                "quantity": product.quantity,
                "link": product.link,
                "image_reference": product.image_reference,
                "asset_id": str(product.asset_id) if product.asset_id else None,
            },
            source={
                "doc_id": str(product.source_doc_id),
                "row_id": str(product.source_row_id) if product.source_row_id else None,
            },
            canonical_key=f"product:{product.product_id}",
        )

    def _service_result(self, session: Session, service: Service, score: float) -> RetrievalResult:
        asset_token = self._asset_token(session, service.asset_id)
        fields = [
            f"Dịch vụ: {service.name}",
            f"Danh mục: {service.category_code}" if service.category_code else None,
            f"Mô tả: {service.description}" if service.description else None,
            f"Thời lượng: {service.duration_minutes} phút"
            if service.duration_minutes is not None
            else None,
            f"Giá: {service.price} {service.currency}"
            if service.price is not None
            else None,
            f"Triệu chứng/chỉ định: {', '.join(service.symptoms)}" if service.symptoms else None,
            f"Ảnh: {asset_token}" if asset_token else None,
        ]
        return RetrievalResult(
            source_type="service",
            source_id=str(service.service_id),
            text=". ".join(field for field in fields if field),
            score=score,
            raw_json={
                "name": service.name,
                "category_code": service.category_code,
                "source_category": service.source_category,
                "description": service.description,
                "duration_minutes": service.duration_minutes,
                "price": serialize_value(service.price),
                "currency": service.currency,
                "symptoms": service.symptoms,
                "indications": service.indications,
                "contraindications": service.contraindications,
                "image_reference": service.image_reference,
                "asset_id": str(service.asset_id) if service.asset_id else None,
            },
            source={
                "doc_id": str(service.source_doc_id),
                "row_id": str(service.source_row_id) if service.source_row_id else None,
            },
            canonical_key=f"service:{service.service_id}",
        )

    @staticmethod
    def _faq_result(faq: FAQ, score: float) -> RetrievalResult:
        return RetrievalResult(
            source_type="faq",
            source_id=str(faq.faq_id),
            text=f"Câu hỏi: {faq.question}\nTrả lời: {faq.answer}",
            score=score,
            raw_json={"question": faq.question, "answer": faq.answer, "category": faq.category},
            canonical_key=f"faq:{faq.faq_id}",
        )

    @staticmethod
    def _asset_token(session: Session, asset_id) -> str | None:
        if asset_id is None:
            return None
        return session.scalar(
            select(Asset.asset_token).where(Asset.asset_id == asset_id, Asset.status == "active")
        )
