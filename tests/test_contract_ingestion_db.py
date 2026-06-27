from dataclasses import fields
from inspect import signature
from typing import Any, get_type_hints

from pgvector.sqlalchemy import Vector

from app.db.models import Chunk, Document, IngestionRun, TableRow
from app.ingestion.chunker import DocumentChunker, TextChunk
from app.ingestion.embedder import EmbeddingService
from app.ingestion.pipeline import IngestionPipeline


def test_chunker_to_db_chunk_contract():
    assert [field.name for field in fields(TextChunk)] == [
        "content",
        "page_number",
        "section_title",
        "content_type",
        "metadata",
    ]
    assert get_type_hints(TextChunk) == {
        "content": str,
        "page_number": int | None,
        "section_title": str | None,
        "content_type": str,
        "metadata": dict[str, Any] | None,
    }
    hints = get_type_hints(DocumentChunker.split)
    assert hints["return"] == list[TextChunk]
    assert isinstance(Chunk.__table__.c.embedding.type, Vector)
    assert Chunk.__table__.c.content.nullable is False
    assert Chunk.__table__.c.doc_id.nullable is False
    assert Chunk.__table__.c.chunk_index.nullable is False


def test_ingestion_db_write_models_and_methods_contract():
    assert {"doc_id", "file_name", "checksum", "status", "metadata"}.issubset(
        set(Document.__table__.columns.keys())
    )
    assert {"run_id", "doc_id", "status", "total_chunks", "total_embeddings"}.issubset(
        set(IngestionRun.__table__.columns.keys())
    )
    assert isinstance(TableRow.__table__.c.embedding.type, Vector)

    assert get_type_hints(EmbeddingService.embed_documents)["return"] == list[list[float]]
    ingest_hints = get_type_hints(IngestionPipeline.ingest)
    assert ingest_hints["return"] == tuple[Document, IngestionRun]
    assert list(signature(IngestionPipeline.ingest).parameters) == [
        "self",
        "session",
        "file_path",
        "options",
    ]
