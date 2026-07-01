from decimal import Decimal
from typing import Any

from rapidfuzz import fuzz
from sqlalchemy import asc, desc, func, or_, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.constants import Intent
from app.db.models import FAQ, Asset, ClinicInfo, Document, FAQAlias, Product, Service
from app.orchestration.intent_registry import EntityScope, capability_for
from app.retrieval.normalization import (
    normalize_vietnamese,
    search_query_tokens,
)
from app.retrieval.structured_query import (
    ProductQuerySpec,
    ServiceQuerySpec,
    parse_product_query,
    parse_service_query,
)
from app.retrieval.types import RetrievalResult


def serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


class StructuredRetriever:
    def __init__(self, settings: Settings | None = None):
        self.match_threshold = (
            settings.entity_match_threshold if settings else 0.72
        )
        self.ambiguity_margin = (
            settings.entity_ambiguity_margin if settings else 0.05
        )

    def active_names(self, session: Session) -> tuple[list[str], list[str]]:
        products = list(
            session.scalars(
                select(Product.name)
                .join(Document, Document.doc_id == Product.source_doc_id)
                .where(Product.status == "active")
                .where(Document.status == "active")
                .distinct()
            ).all()
        )
        services = list(
            session.scalars(
                select(Service.name)
                .join(Document, Document.doc_id == Service.source_doc_id)
                .where(Service.status == "active")
                .where(Document.status == "active")
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
        capability = capability_for(intent)
        domain = capability.entity_domain
        scope = capability.entity_scope
        if domain == "product" and scope == EntityScope.FILTER_ONLY:
            specification = parse_product_query(session, query)
            if specification.needs_clarification:
                return []
            return [
                self._product_result(session, item, 1.0)
                for item in self.list_products(session, specification)
            ]
        if domain == "service" and scope == EntityScope.FILTER_ONLY:
            specification = parse_service_query(session, query)
            if specification.needs_clarification:
                return []
            return [
                self._service_result(session, item, 1.0)
                for item in self.list_services(session, specification)
            ]
        if domain == "product" and scope == EntityScope.EXACTLY_ONE:
            product = self.get_product(session, entities[0] if entities else query)
            return [self._product_result(session, product[0], product[1])] if product else []
        if domain == "product" and scope == EntityScope.TWO_OR_MORE:
            results = []
            for entity in entities:
                product = self.get_product(session, entity)
                if product:
                    results.append(self._product_result(session, product[0], product[1]))
            return results
        if domain == "service" and scope == EntityScope.EXACTLY_ONE:
            service = self.get_service(session, entities[0] if entities else query)
            return [self._service_result(session, service[0], service[1])] if service else []
        if capability.primary_source_type == "clinic_info":
            return self.clinic_info(session, query)
        if capability.primary_source_type == "faq":
            faq = self.get_faq(session, query)
            return [self._faq_result(faq[0], faq[1])] if faq else []
        return []

    def list_products(
        self,
        session: Session,
        specification: ProductQuerySpec | None = None,
    ) -> list[Product]:
        specification = specification or ProductQuerySpec()
        statement = (
            select(Product)
            .join(Document, Document.doc_id == Product.source_doc_id)
            .where(Product.status == "active", Document.status == "active")
        )
        if specification.category_codes:
            statement = statement.where(
                Product.category_code.in_(specification.category_codes)
            )
        if specification.product_ids:
            statement = statement.where(
                Product.product_id.in_(specification.product_ids)
            )
        if specification.brand_terms:
            statement = statement.where(
                self._product_terms_clause(
                    specification.brand_terms,
                    Product.brand,
                    Product.name,
                    Product.model,
                )
            )
        if specification.feature_terms:
            statement = statement.where(
                self._product_terms_clause(
                    specification.feature_terms,
                    Product.name,
                    Product.category,
                    Product.brand,
                    Product.model,
                    Product.description,
                )
            )
        if specification.price_min is not None:
            statement = statement.where(Product.price >= specification.price_min)
        if specification.price_max is not None:
            statement = statement.where(Product.price <= specification.price_max)
        if specification.quantity_min is not None:
            statement = statement.where(Product.quantity >= specification.quantity_min)
        if specification.quantity_max is not None:
            statement = statement.where(Product.quantity <= specification.quantity_max)
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

    def list_services(
        self,
        session: Session,
        specification: ServiceQuerySpec | None = None,
    ) -> list[Service]:
        specification = specification or ServiceQuerySpec()
        statement = (
            select(Service)
            .join(Document, Document.doc_id == Service.source_doc_id)
            .where(Service.status == "active", Document.status == "active")
        )
        if specification.category_codes or specification.category_terms:
            category_clauses = []
            if specification.category_codes:
                category_clauses.append(
                    Service.category_code.in_(specification.category_codes)
                )
            if specification.category_terms:
                category_clauses.append(
                    self._service_terms_clause(
                        specification.category_terms,
                        Service.source_category,
                        Service.category_code,
                        Service.name,
                    )
                )
            if category_clauses:
                statement = statement.where(or_(*category_clauses))
        if specification.service_ids:
            statement = statement.where(Service.service_id.in_(specification.service_ids))
        if specification.price_min is not None:
            statement = statement.where(Service.price >= specification.price_min)
        if specification.price_max is not None:
            statement = statement.where(Service.price <= specification.price_max)
        if specification.feature_terms:
            statement = statement.where(
                self._service_terms_clause(
                    specification.feature_terms,
                    Service.name,
                    Service.source_category,
                    Service.category_code,
                    Service.description,
                    func.array_to_string(Service.symptoms, " "),
                    func.array_to_string(Service.indications, " "),
                    func.array_to_string(Service.contraindications, " "),
                )
            )
        if specification.symptom_terms:
            statement = statement.where(
                self._service_terms_clause(
                    specification.symptom_terms,
                    Service.name,
                    Service.description,
                    func.array_to_string(Service.symptoms, " "),
                    func.array_to_string(Service.indications, " "),
                    func.array_to_string(Service.contraindications, " "),
                )
            )
        if specification.duration_min is not None:
            statement = statement.where(
                Service.duration_minutes >= specification.duration_min
            )
        if specification.duration_max is not None:
            statement = statement.where(
                Service.duration_minutes <= specification.duration_max
            )
        sort_columns = {
            "price": Service.price,
            "duration": Service.duration_minutes,
            "name": Service.name,
            "category": Service.source_category,
        }
        sort_column = sort_columns.get(specification.sort_by, Service.name)
        ordering = (
            desc(sort_column)
            if specification.sort_direction == "desc"
            else asc(sort_column)
        )
        return list(
            session.scalars(
                statement.order_by(ordering.nulls_last(), Service.name)
                .limit(specification.limit)
            ).all()
        )

    def search_faqs(
        self,
        session: Session,
        query: str,
        *,
        limit: int = 5,
    ) -> list[tuple[FAQ, float]]:
        faqs = list(
            session.scalars(
                select(FAQ)
                .outerjoin(Document, Document.doc_id == FAQ.source_doc_id)
                .where(
                    FAQ.is_active.is_(True),
                    or_(FAQ.source_doc_id.is_(None), Document.status == "active"),
                )
            ).all()
        )
        if not faqs:
            return []
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
            score = max(self._faq_match_score(query, candidate) for candidate in candidates)
            if score >= 0.72:
                scored.append((faq, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def get_faq(self, session: Session, query: str) -> tuple[FAQ, float] | None:
        results = self.search_faqs(session, query, limit=1)
        return results[0] if results else None

    def get_product(self, session: Session, name_or_query: str) -> tuple[Product, float] | None:
        products = self.list_products(session)
        return self._best_match(products, name_or_query)

    def get_service(self, session: Session, name_or_query: str) -> tuple[Service, float] | None:
        services = self.list_services(session)
        return self._best_match(services, name_or_query)

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
                select(ClinicInfo)
                .outerjoin(Document, Document.doc_id == ClinicInfo.source_doc_id)
                .where(
                    ClinicInfo.status == "active",
                    or_(
                        ClinicInfo.source_doc_id.is_(None),
                        Document.status == "active",
                    ),
                )
                .order_by(ClinicInfo.key)
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

    def _best_match(self, records: list[Any], query: str) -> tuple[Any, float] | None:
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
            if (
                normalize_vietnamese(record.name) in normalized
                or normalized in normalize_vietnamese(record.name)
            )
        ]
        if contained:
            return min(
                contained,
                key=lambda item: abs(len(normalize_vietnamese(item.name)) - len(normalized)),
            ), 0.96

        query_tokens = self._entity_tokens(query)
        scored = [
            (
                record,
                self._entity_match_score(
                    normalized,
                    query_tokens,
                    normalize_vietnamese(record.name),
                    self._entity_tokens(record.name),
                ),
            )
            for record in records
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best = scored[0]
        if best[1] < self.match_threshold:
            return None
        if len(scored) > 1 and best[1] - scored[1][1] < self.ambiguity_margin:
            return None
        return best

    @staticmethod
    def _entity_tokens(value: str) -> set[str]:
        stopwords = {
            "toi",
            "muon",
            "hoi",
            "ve",
            "cho",
            "xem",
            "dich",
            "vu",
            "san",
            "pham",
            "gia",
            "chi",
            "phi",
            "bao",
            "nhieu",
            "co",
            "khong",
            "mat",
            "lau",
            "thoi",
            "gian",
            "la",
            "cua",
        }
        return {
            token
            for token in search_query_tokens(value)
            if token not in stopwords and len(token) > 1
        }

    @staticmethod
    def _entity_match_score(
        normalized_query: str,
        query_tokens: set[str],
        normalized_name: str,
        name_tokens: set[str],
    ) -> float:
        fuzzy_score = fuzz.WRatio(normalized_query, normalized_name) / 100.0
        overlap = query_tokens & name_tokens
        query_coverage = len(overlap) / max(1, len(query_tokens))
        name_coverage = len(overlap) / max(1, len(name_tokens))
        if len(query_tokens) >= 2 and query_coverage == 1.0:
            return min(0.98, 0.94 + 0.04 * name_coverage)
        return 0.55 * fuzzy_score + 0.30 * query_coverage + 0.15 * name_coverage

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
        category = service.source_category or service.category_code
        fields = [
            f"Dịch vụ: {service.name}",
            f"Danh mục: {category}" if category else None,
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
    def _service_matches_category(
        service: Service,
        specification: ServiceQuerySpec,
    ) -> bool:
        if service.category_code in specification.category_codes:
            return True
        haystack = normalize_vietnamese(
            " ".join(
                value
                for value in (
                    service.source_category,
                    service.category_code,
                    service.name,
                )
                if value
            )
        )
        return any(term and term in haystack for term in specification.category_terms)

    @staticmethod
    def _service_matches_terms(service: Service, terms: tuple[str, ...]) -> bool:
        haystack = normalize_vietnamese(
            " ".join(
                value
                for value in (
                    service.name,
                    service.source_category,
                    service.category_code,
                    service.description,
                    " ".join(service.symptoms or []),
                    " ".join(service.indications or []),
                    " ".join(service.contraindications or []),
                )
                if value
            )
        )
        return any(normalize_vietnamese(term) in haystack for term in terms if term)

    @staticmethod
    def _product_terms_clause(terms: tuple[str, ...], *columns):
        clauses = []
        normalized_terms = [
            normalize_vietnamese(term)
            for term in terms
            if str(term or "").strip()
        ]
        for term in normalized_terms:
            for column in columns:
                clauses.append(
                    func.lower(func.simplydent_unaccent(column)).contains(term)
                )
        return or_(*clauses) if clauses else True

    @staticmethod
    def _service_terms_clause(terms: tuple[str, ...], *columns):
        return StructuredRetriever._product_terms_clause(terms, *columns)

    @staticmethod
    def _service_sort_value(service: Service, sort_by: str):
        if sort_by == "price":
            return service.price
        if sort_by == "duration":
            return service.duration_minutes
        if sort_by == "category":
            return service.source_category or service.category_code
        return service.name

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
