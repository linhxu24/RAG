"""Smoke-check ingestion writes chunks and embeddings to PostgreSQL."""

import tempfile
from pathlib import Path

from sqlalchemy import select

from app.admin.data_reset import delete_document
from app.config import get_settings
from app.db.models import Chunk
from app.db.session import get_session_factory
from app.ingestion.pipeline import IngestionOptions, IngestionPipeline


def main() -> None:
    settings = get_settings().model_copy(
        update={
            "allow_embedding_fallback": True,
            "strict_embedding": False,
        }
    )
    with tempfile.TemporaryDirectory(prefix="simplydent-ingestion-smoke-") as directory:
        source = Path(directory) / "smoke.txt"
        source.write_text(
            "AquaJet Mini Water Flosser là sản phẩm nha khoa dùng để vệ sinh răng.",
            encoding="utf-8",
        )
        with get_session_factory()() as session:
            document, run = IngestionPipeline(settings).ingest(
                session,
                source,
                IngestionOptions(
                    document_type="auto",
                    create_embeddings=True,
                    duplicate_policy="force",
                    original_file_name=source.name,
                ),
            )
            try:
                chunks = session.scalars(
                    select(Chunk).where(Chunk.doc_id == document.doc_id)
                ).all()
                assert run.status == "completed"
                assert chunks
                assert all(chunk.embedding is not None for chunk in chunks)
                assert (
                    document.metadata_json
                    and "ingestion_entity_extraction" in document.metadata_json
                )
                print(
                    {
                        "doc_id": str(document.doc_id),
                        "chunks": len(chunks),
                        "embedding_dimensions": len(chunks[0].embedding),
                        "entity_mentions": document.metadata_json[
                            "ingestion_entity_extraction"
                        ]["total_mentions"],
                    }
                )
            finally:
                delete_document(session, settings, document.doc_id)


if __name__ == "__main__":
    main()
