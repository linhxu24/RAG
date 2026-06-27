"""Smoke-check database bootstrap requirements used by ORM/retrieval."""

from sqlalchemy import text

from app.db.init_db import init_database
from app.db.session import get_session_factory


def main() -> None:
    init_database()
    with get_session_factory()() as session:
        capabilities = session.execute(
            text(
                """
                SELECT
                  EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                  ) AS vector_enabled,
                  EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'
                  ) AS trigram_enabled,
                  EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'unaccent'
                  ) AS unaccent_enabled,
                  to_regprocedure('simplydent_unaccent(text)') IS NOT NULL
                    AS unaccent_function_enabled
                """
            )
        ).mappings().one()
    missing = [name for name, enabled in capabilities.items() if not enabled]
    if missing:
        raise SystemExit(f"Missing DB bootstrap capabilities: {missing}")
    print(dict(capabilities))


if __name__ == "__main__":
    main()
