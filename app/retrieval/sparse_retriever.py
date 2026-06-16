from sqlalchemy import and_, func, literal, or_, select
from sqlalchemy.orm import Session

from app.constants import Intent
from app.db.models import FAQ, Chunk, Product, Service, TableRow
from app.retrieval.normalization import normalize_vietnamese, search_query_tokens
from app.retrieval.types import RetrievalResult


class SparseRetriever:
    def __init__(
        self,
        top_k: int = 20,
        trigram_threshold: float = 0.2,
        min_fts_rank: float = 0.001,
        max_per_source: int = 10,
    ):
        self.top_k = top_k
        self.trigram_threshold = trigram_threshold
        self.min_fts_rank = min_fts_rank
        self.max_per_source = max_per_source

    def retrieve(self, session: Session, query: str, intent: Intent) -> list[RetrievalResult]:
        result_sets = self.retrieve_by_source(session, query, intent)
        results = [item for values in result_sets.values() for item in values]
        return sorted(results, key=lambda item: item.score, reverse=True)

    def retrieve_by_source(
        self,
        session: Session,
        query: str,
        intent: Intent,
    ) -> dict[str, list[RetrievalResult]]:
        normalized_query = normalize_vietnamese(query)
        tokens = search_query_tokens(query)
        if not tokens:
            return {}
        tsquery = func.to_tsquery(
            "simple",
            " | ".join(f"{token}:*" for token in tokens),
        )
        if intent == Intent.FAQ:
            return {
                "faq": self._faqs(
                    session,
                    tsquery,
                    normalized_query,
                )
            }
        result_sets = {
            "chunk": self._chunks(session, tsquery, normalized_query),
            "table_row": self._rows(
                session,
                tsquery,
                normalized_query,
                intent,
            ),
        }
        if intent == Intent.UNKNOWN:
            result_sets["faq"] = self._faqs(
                session,
                tsquery,
                normalized_query,
            )
        return result_sets

    def _chunks(
        self,
        session: Session,
        tsquery,
        normalized_query: str,
    ) -> list[RetrievalResult]:
        rank = func.ts_rank_cd(Chunk.content_tsv, tsquery)
        rows = session.execute(
            select(Chunk, rank.label("rank"))
            .where(Chunk.status == "active", Chunk.content_tsv.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(self.top_k)
        ).all()
        results = [
            RetrievalResult(
                source_type="chunk",
                source_id=str(chunk.chunk_id),
                text=chunk.content,
                score=float(score),
                source={"doc_id": str(chunk.doc_id), "page_number": chunk.page_number},
                canonical_key=f"chunk:{chunk.chunk_id}",
            )
            for chunk, score in rows
            if float(score) >= self.min_fts_rank
        ]
        if results:
            return results[: self.max_per_source]
        return self._chunk_trigram(session, normalized_query)[: self.max_per_source]

    def _rows(
        self,
        session: Session,
        tsquery,
        normalized_query: str,
        intent: Intent,
    ) -> list[RetrievalResult]:
        rank = func.ts_rank_cd(TableRow.row_tsv, tsquery)
        statement = (
            select(
                TableRow,
                rank.label("rank"),
                Product.product_id,
                Service.service_id,
                FAQ.faq_id,
            )
            .outerjoin(
                Product,
                and_(
                    Product.source_row_id == TableRow.row_id,
                    Product.status == "active",
                ),
            )
            .outerjoin(
                Service,
                and_(
                    Service.source_row_id == TableRow.row_id,
                    Service.status == "active",
                ),
            )
            .outerjoin(
                FAQ,
                and_(
                    FAQ.is_active.is_(True),
                    FAQ.source_row_id == TableRow.row_id,
                ),
            )
            .where(
                TableRow.status == "active",
                TableRow.row_tsv.op("@@")(tsquery),
            )
            .order_by(rank.desc())
            .limit(self.top_k)
        )
        entity_type = self._entity_type_filter(intent)
        if entity_type:
            statement = statement.where(
                or_(
                    TableRow.entity_type == entity_type,
                    TableRow.entity_type.is_(None),
                )
            )
        rows = session.execute(statement).all()
        results = [
            RetrievalResult(
                source_type="table_row",
                source_id=str(row.row_id),
                text=row.row_text,
                score=float(score),
                raw_json=row.row_json,
                source={
                    "doc_id": str(row.doc_id),
                    "table_id": str(row.table_id),
                    "entity_type": row.entity_type,
                    "entity_name": row.entity_name,
                },
                canonical_key=self._row_canonical_key(
                    row,
                    product_id,
                    service_id,
                    faq_id,
                ),
            )
            for row, score, product_id, service_id, faq_id in rows
            if float(score) >= self.min_fts_rank
        ]
        if results:
            return results[: self.max_per_source]
        return self._row_trigram(session, normalized_query, intent)[
            : self.max_per_source
        ]

    def _faqs(
        self,
        session: Session,
        tsquery,
        normalized_query: str,
    ) -> list[RetrievalResult]:
        rank = func.ts_rank_cd(FAQ.question_tsv, tsquery)
        rows = session.execute(
            select(FAQ, rank.label("rank"))
            .where(FAQ.is_active.is_(True), FAQ.question_tsv.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(self.top_k)
        ).all()
        results = [
            RetrievalResult(
                source_type="faq",
                source_id=str(faq.faq_id),
                text=f"Câu hỏi: {faq.question}\nTrả lời: {faq.answer}",
                score=float(score),
                raw_json={"question": faq.question, "answer": faq.answer},
                canonical_key=f"faq:{faq.faq_id}",
            )
            for faq, score in rows
            if float(score) >= self.min_fts_rank
        ]
        if results:
            return results[: self.max_per_source]
        return self._faq_trigram(session, normalized_query)[: self.max_per_source]

    def _chunk_trigram(
        self,
        session: Session,
        normalized_query: str,
    ) -> list[RetrievalResult]:
        normalized_text = func.lower(func.simplydent_unaccent(Chunk.content))
        score = func.similarity(normalized_text, literal(normalized_query))
        rows = session.execute(
            select(Chunk, score.label("score"))
            .where(
                Chunk.status == "active",
                score >= self.trigram_threshold,
            )
            .order_by(score.desc())
            .limit(self.top_k)
        ).all()
        return [
            RetrievalResult(
                source_type="chunk",
                source_id=str(chunk.chunk_id),
                text=chunk.content,
                score=float(value),
                source={
                    "doc_id": str(chunk.doc_id),
                    "page_number": chunk.page_number,
                },
                canonical_key=f"chunk:{chunk.chunk_id}",
            )
            for chunk, value in rows
        ]

    def _row_trigram(
        self,
        session: Session,
        normalized_query: str,
        intent: Intent,
    ) -> list[RetrievalResult]:
        normalized_text = func.lower(func.simplydent_unaccent(TableRow.row_text))
        score = func.similarity(normalized_text, literal(normalized_query))
        statement = (
            select(
                TableRow,
                score.label("score"),
                Product.product_id,
                Service.service_id,
                FAQ.faq_id,
            )
            .outerjoin(
                Product,
                and_(
                    Product.source_row_id == TableRow.row_id,
                    Product.status == "active",
                ),
            )
            .outerjoin(
                Service,
                and_(
                    Service.source_row_id == TableRow.row_id,
                    Service.status == "active",
                ),
            )
            .outerjoin(
                FAQ,
                and_(
                    FAQ.is_active.is_(True),
                    FAQ.source_row_id == TableRow.row_id,
                ),
            )
            .where(
                TableRow.status == "active",
                score >= self.trigram_threshold,
            )
            .order_by(score.desc())
            .limit(self.top_k)
        )
        entity_type = self._entity_type_filter(intent)
        if entity_type:
            statement = statement.where(
                or_(
                    TableRow.entity_type == entity_type,
                    TableRow.entity_type.is_(None),
                )
            )
        rows = session.execute(statement).all()
        return [
            RetrievalResult(
                source_type="table_row",
                source_id=str(row.row_id),
                text=row.row_text,
                score=float(value),
                raw_json=row.row_json,
                source={
                    "doc_id": str(row.doc_id),
                    "table_id": str(row.table_id),
                    "entity_type": row.entity_type,
                    "entity_name": row.entity_name,
                },
                canonical_key=self._row_canonical_key(
                    row,
                    product_id,
                    service_id,
                    faq_id,
                ),
            )
            for row, value, product_id, service_id, faq_id in rows
        ]

    def _faq_trigram(
        self,
        session: Session,
        normalized_query: str,
    ) -> list[RetrievalResult]:
        faq_text = FAQ.question + literal(" ") + FAQ.answer
        normalized_text = func.lower(func.simplydent_unaccent(faq_text))
        score = func.similarity(normalized_text, literal(normalized_query))
        rows = session.execute(
            select(FAQ, score.label("score"))
            .where(
                FAQ.is_active.is_(True),
                score >= self.trigram_threshold,
            )
            .order_by(score.desc())
            .limit(self.top_k)
        ).all()
        return [
            RetrievalResult(
                source_type="faq",
                source_id=str(faq.faq_id),
                text=f"Câu hỏi: {faq.question}\nTrả lời: {faq.answer}",
                score=float(value),
                raw_json={"question": faq.question, "answer": faq.answer},
                canonical_key=f"faq:{faq.faq_id}",
            )
            for faq, value in rows
        ]

    @staticmethod
    def _entity_type_filter(intent: Intent) -> str | None:
        if intent in {Intent.PRODUCT_DETAIL, Intent.PRODUCT_COMPARE}:
            return "product"
        if intent == Intent.SERVICE_DETAIL:
            return "service"
        if intent == Intent.FAQ:
            return "faq"
        return None

    @staticmethod
    def _row_canonical_key(row, product_id, service_id, faq_id) -> str:
        if product_id:
            return f"product:{product_id}"
        if service_id:
            return f"service:{service_id}"
        if faq_id:
            return f"faq:{faq_id}"
        return f"table_row:{row.row_id}"
