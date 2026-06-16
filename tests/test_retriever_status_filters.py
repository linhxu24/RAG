from sqlalchemy import func

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever


class EmptyRows:
    def all(self):
        return []


class CapturingSession:
    def __init__(self):
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return EmptyRows()


def test_dense_chunk_query_filters_active_status():
    session = CapturingSession()
    retriever = DenseRetriever(embedder=object())

    retriever._chunks(session, [0.0] * 1024)

    sql = str(session.statement)
    assert "chunks.status" in sql
    assert "chunks.embedding IS NOT NULL" in sql


def test_sparse_faq_query_filters_is_active():
    session = CapturingSession()
    retriever = SparseRetriever()
    tsquery = func.to_tsquery("simple", "implant:*")

    retriever._faqs(session, tsquery, "implant")

    sql = str(session.statement)
    assert "faqs.is_active" in sql
