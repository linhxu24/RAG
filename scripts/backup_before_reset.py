"""Create a PostgreSQL custom-format dump and a content-count snapshot."""

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from app.admin.data_reset import data_counts
from app.config import get_settings
from app.db.session import get_session_factory


def main() -> None:
    settings = get_settings()
    backup_dir = Path("backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dump_path = backup_dir / f"simplydent_pre_reset_{timestamp}.dump"
    stats_path = backup_dir / f"simplydent_pre_reset_{timestamp}.json"

    url = make_url(settings.resolved_database_url)
    pg_dump = shutil.which("pg_dump") or (
        "/Applications/Postgres.app/Contents/Versions/latest/bin/pg_dump"
    )
    if not Path(pg_dump).is_file():
        raise RuntimeError("pg_dump was not found")
    environment = os.environ.copy()
    if url.password:
        environment["PGPASSWORD"] = url.password
    command = [
        pg_dump,
        "--format=custom",
        "--no-owner",
        "--no-privileges",
        f"--host={url.host or 'localhost'}",
        f"--port={url.port or 5432}",
        f"--username={url.username or 'postgres'}",
        f"--file={dump_path}",
        str(url.database),
    ]
    subprocess.run(command, check=True, env=environment)

    with get_session_factory()() as session:
        revision = session.scalar(select(text("version_num")).select_from(text("alembic_version")))
        snapshot = {
            "created_at": datetime.now(UTC).isoformat(),
            "database": str(url.database),
            "alembic_revision": revision,
            "counts": data_counts(session),
            "dump_file": str(dump_path),
            "dump_bytes": dump_path.stat().st_size,
        }
    stats_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print({"dump": str(dump_path), "stats": str(stats_path), **snapshot})


if __name__ == "__main__":
    main()
