"""Smoke-check structured service query resolves service aliases."""

from app.config import get_settings
from app.db.models import Document, Service, ServiceAlias
from app.db.session import get_session_factory
from app.retrieval.normalization import normalize_vietnamese
from app.retrieval.structured_query import parse_service_query


def main() -> None:
    with get_session_factory()() as session:
        document = Document(
            file_name="smoke-service-alias.csv",
            checksum="smoke-service-alias",
            status="active",
        )
        session.add(document)
        session.flush()
        service = Service(
            name="Điều trị nha chu chuyên sâu",
            source_doc_id=document.doc_id,
            status="active",
        )
        session.add(service)
        session.flush()
        session.add(
            ServiceAlias(
                service_id=service.service_id,
                alias="chữa viêm nướu",
                normalized_alias=normalize_vietnamese("chữa viêm nướu"),
            )
        )
        session.commit()
        try:
            spec = parse_service_query(
                session,
                "Tôi muốn xem dịch vụ chữa viêm nướu",
            )
            assert str(service.service_id) in spec.service_ids
            print(spec.as_dict())
        finally:
            session.delete(document)
            session.commit()


if __name__ == "__main__":
    get_settings()
    main()
