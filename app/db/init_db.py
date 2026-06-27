from sqlalchemy import text

from app.db.models import Base
from app.db.session import get_engine


def init_database() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
        connection.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION simplydent_unaccent(value text)
                RETURNS text
                LANGUAGE sql
                IMMUTABLE
                PARALLEL SAFE
                RETURNS NULL ON NULL INPUT
                AS $$ SELECT public.unaccent(value) $$
                """
            )
        )
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    init_database()
