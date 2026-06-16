from unittest.mock import Mock

from app.config import Settings
from app.constants import Intent
from app.retrieval.entity_resolver import (
    DatabaseEntityResolver,
    EntityCandidate,
)


def test_duplicate_database_rows_with_same_name_are_not_ambiguous():
    resolver = DatabaseEntityResolver(Settings())
    resolver._candidates = Mock(
        return_value=[
            EntityCandidate(
                entity_type="service",
                entity_id="service-1",
                name="Nhổ răng khôn",
                score=1.0,
                match_type="contained",
            ),
            EntityCandidate(
                entity_type="service",
                entity_id="service-2",
                name="Nhổ răng khôn",
                score=1.0,
                match_type="contained",
            ),
        ]
    )

    result = resolver.resolve(
        Mock(),
        "Chi phí nhổ răng khôn là bao nhiêu?",
        Intent.SERVICE_DETAIL,
    )

    assert result.status == "resolved"
    assert result.names == ["Nhổ răng khôn"]
    assert result.ambiguous_candidates == []
    assert [item.name for item in result.candidates] == ["Nhổ răng khôn"]
