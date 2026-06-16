import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.session import check_database


@pytest.mark.integration
def test_postgresql_connection_and_pgvector():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run the PostgreSQL integration test")
    with Session(create_engine(url)) as session:
        result = check_database(session)
    assert result["connected"] is True
    assert result["pgvector_enabled"] is True
