from inspect import signature
from typing import get_type_hints

from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import Intent
from app.db.models import FAQ, Chunk, Document, Product, Service, TableRow
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.structured_retriever import StructuredRetriever
from app.retrieval.types import RetrievalResult


def _column_names(model):
    return [column.name for column in model.__table__.columns]


def test_retrieval_reads_required_db_columns_and_vector_dimensions():
    assert _column_names(Document) == [
        "doc_id",
        "file_name",
        "file_type",
        "source_path",
        "uploaded_at",
        "checksum",
        "status",
        "version",
        "detected_document_type",
        "document_type_confidence",
        "metadata",
    ]
    assert {"chunk_id", "doc_id", "content", "content_tsv", "embedding", "status"}.issubset(
        set(_column_names(Chunk))
    )
    assert {"row_id", "doc_id", "row_text", "row_tsv", "embedding", "status"}.issubset(
        set(_column_names(TableRow))
    )
    assert {"faq_id", "question_tsv", "embedding", "is_active"}.issubset(
        set(_column_names(FAQ))
    )
    assert {"product_id", "name", "price", "quantity", "asset_id", "status"}.issubset(
        set(_column_names(Product))
    )
    assert {"service_id", "name", "duration_minutes", "price", "asset_id", "status"}.issubset(
        set(_column_names(Service))
    )
    assert isinstance(Chunk.__table__.c.embedding.type, Vector)
    assert isinstance(TableRow.__table__.c.embedding.type, Vector)
    assert isinstance(FAQ.__table__.c.embedding.type, Vector)
    assert Chunk.__table__.c.embedding.type.dim == get_settings().embedding_dim


def test_retrievers_accept_session_query_intent_and_return_retrieval_results():
    for retriever in (DenseRetriever, SparseRetriever):
        hints = get_type_hints(retriever.retrieve)
        assert hints["session"] is Session
        assert hints["query"] is str
        assert hints["intent"] is Intent
        assert hints["return"] == list[RetrievalResult]
        assert list(signature(retriever.retrieve).parameters) == [
            "self",
            "session",
            "query",
            "intent",
        ]

    hints = get_type_hints(StructuredRetriever.retrieve)
    assert hints["session"] is Session
    assert hints["intent"] is Intent
    assert hints["query"] is str
    assert hints["entities"] == list[str]
    assert hints["return"] == list[RetrievalResult]
