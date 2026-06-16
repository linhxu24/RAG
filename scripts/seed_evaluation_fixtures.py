from pathlib import Path

from app.config import get_settings
from app.db.session import get_session_factory
from app.ingestion.pipeline import IngestionOptions, IngestionPipeline

FIXTURES = (
    ("eval_datasets/fixtures/dental_services_v1.csv", "service"),
    ("eval_datasets/fixtures/dental_faqs_v1.csv", "faq"),
    ("eval_datasets/fixtures/dental_clinic_info_v1.csv", "clinic_info"),
)


def main() -> None:
    settings = get_settings()
    pipeline = IngestionPipeline(settings)
    with get_session_factory()() as session:
        for source, document_type in FIXTURES:
            document, run = pipeline.ingest(
                session,
                Path(source),
                IngestionOptions(
                    document_type=document_type,
                    duplicate_policy="replace",
                ),
            )
            print(
                {
                    "source": source,
                    "doc_id": str(document.doc_id),
                    "status": document.status,
                    "run_id": str(run.run_id),
                    "smoke_passed": run.quality_report.get("smoke_test", {}).get("passed"),
                }
            )


if __name__ == "__main__":
    main()
