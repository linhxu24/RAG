from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.constants import Intent
from app.db.models import FAQ, Chunk, Product, Service, TableRow
from app.ingestion.embedder import EmbeddingService
from app.retrieval.types import RetrievalResult


class DenseRetriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        top_k: int = 20,
        min_score: float = 0.25,
    ):
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

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
        vector = self.embedder.embed_query(query)
        if intent == Intent.FAQ:
            return {"faq": self._faqs(session, vector)}
        result_sets = {
            "chunk": self._chunks(session, vector),
            "table_row": self._rows(session, vector, intent),
        }
        if intent == Intent.UNKNOWN:
            result_sets["faq"] = self._faqs(session, vector)
        return result_sets

    def _chunks(self, session: Session, vector: list[float]) -> list[RetrievalResult]:
        distance = Chunk.embedding.cosine_distance(vector)
        rows = session.execute(
            select(Chunk, distance.label("distance"))
            .where(Chunk.status == "active", Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(self.top_k)
        ).all()
        return self._above_threshold([
            RetrievalResult(
                source_type="chunk",
                source_id=str(chunk.chunk_id),
                text=chunk.content,
                score=1.0 - float(distance_value),
                source={
                    "doc_id": str(chunk.doc_id),
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                },
                canonical_key=f"chunk:{chunk.chunk_id}",
            )
            for chunk, distance_value in rows
        ])

    def _rows(
        self,
        session: Session,
        vector: list[float],
        intent: Intent,
    ) -> list[RetrievalResult]:
        distance = TableRow.embedding.cosine_distance(vector)
        statement = (
            select(
                TableRow,
                distance.label("distance"),
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
            .where(TableRow.status == "active", TableRow.embedding.is_not(None))
            .order_by(distance)
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
        return self._above_threshold([
            RetrievalResult(
                source_type="table_row",
                source_id=str(row.row_id),
                text=row.row_text,
                score=1.0 - float(distance_value),
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
            for row, distance_value, product_id, service_id, faq_id in rows
        ])

    def _faqs(self, session: Session, vector: list[float]) -> list[RetrievalResult]:
        distance = FAQ.embedding.cosine_distance(vector)
        rows = session.execute(
            select(FAQ, distance.label("distance"))
            .where(FAQ.is_active.is_(True), FAQ.embedding.is_not(None))
            .order_by(distance)
            .limit(self.top_k)
        ).all()
        return self._above_threshold([
            RetrievalResult(
                source_type="faq",
                source_id=str(faq.faq_id),
                text=f"Câu hỏi: {faq.question}\nTrả lời: {faq.answer}",
                score=1.0 - float(distance_value),
                raw_json={"question": faq.question, "answer": faq.answer},
                canonical_key=f"faq:{faq.faq_id}",
            )
            for faq, distance_value in rows
        ])

    def _above_threshold(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return [item for item in results if item.score >= self.min_score]

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
