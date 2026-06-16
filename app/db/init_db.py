from sqlalchemy import text

from app.db.models import Base
from app.db.session import get_engine


def init_database() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    init_database()
