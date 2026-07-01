#!/usr/bin/env python3
"""Verify the local Python environment before running Phase 1 tests."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

BASE_IMPORTS = {
    "fastapi": "fastapi",
    "httpx": "httpx",
    "numpy": "numpy",
    "pandas": "pandas",
    "pgvector": "pgvector",
    "pillow": "PIL",
    "psycopg": "psycopg",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "rapidfuzz": "rapidfuzz",
    "sqlalchemy": "sqlalchemy",
    "uvicorn": "uvicorn",
}

DEV_IMPORTS = {
    "pytest": "pytest",
}

DEV_COMMANDS = ("ruff",)

OPTIONAL_IMPORTS = {
    "docling": "docling",
    "openpyxl": "openpyxl",
    "python-docx": "docx",
    "gliner": "gliner",
    "sentence-transformers": "sentence_transformers",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check local dependencies needed to run the backend test suite."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also check ingestion/model optional extras.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    if sys.version_info < (3, 11):  # noqa: UP036
        failures.append(
            f"Python >=3.11 is required; found {sys.version.split()[0]}"
        )

    checks = {
        **BASE_IMPORTS,
        **DEV_IMPORTS,
        **(OPTIONAL_IMPORTS if args.full else {}),
    }
    for package_name, import_name in checks.items():
        if importlib.util.find_spec(import_name) is None:
            failures.append(f"{package_name} is not importable as {import_name}")
    executable_dirs = {
        Path(sys.executable).parent,
        Path(sys.executable).resolve().parent,
        Path(sys.prefix) / "bin",
    }
    for command in DEV_COMMANDS:
        if shutil.which(command) is None and not any(
            (path / command).exists() for path in executable_dirs
        ):
            failures.append(f"{command} is not available on PATH")

    if failures:
        print("Preflight failed:")
        for failure in failures:
            print(f"- {failure}")
        print()
        print("Install dependencies with:")
        print("  uv sync --extra dev --extra ingestion --extra models")
        print()
        print("Then run tests with:")
        print("  uv run pytest")
        return 1

    print("Preflight passed.")
    print("Run tests with: uv run pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
