import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app import __version__
from app.api.routes_admin import router as admin_router
from app.api.routes_chat import (
    get_chat_service,
)
from app.api.routes_chat import (
    router as chat_router,
)
from app.api.routes_control_center import router as control_center_router
from app.api.routes_evaluation import router as evaluation_router
from app.api.routes_ingestion import router as ingestion_router
from app.assets.storage import AssetStorage
from app.config import get_settings
from app.db.models import Asset
from app.db.session import check_database, get_session_factory
from app.observability.logging import configure_logging
from app.orchestration.intent_registry import validate_intent_registry

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_intent_registry()
    settings.ensure_directories()
    configure_logging(settings.debug)
    storage = AssetStorage(settings)
    storage.cleanup_staging()
    with get_session_factory()() as session:
        storage.cleanup_untracked_files(
            set(session.scalars(select(Asset.local_path)).all())
        )
    chat_service = get_chat_service()
    if settings.validate_embedding_on_startup:
        with get_session_factory()() as session:
            chat_service.dense.embedder.validate_configuration(session)
    if settings.enable_reranker and settings.preload_reranker_on_startup:
        chat_service.reranker.warmup()
    if settings.enable_gliner_ner and settings.preload_gliner_on_startup:
        await asyncio.to_thread(chat_service.entity_span_extractor.warmup)
    yield


app = FastAPI(
    title="SimplyDent Multimodal RAG",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
settings.ensure_directories()
app.mount(
    settings.asset_public_base_url,
    StaticFiles(directory=settings.asset_storage_dir),
    name="assets",
)
app.include_router(chat_router)
app.include_router(ingestion_router)
app.include_router(admin_router)
app.include_router(evaluation_router)
app.include_router(control_center_router)


@app.get("/health", tags=["system"])
def health() -> dict:
    database: dict
    try:
        with get_session_factory()() as session:
            database = check_database(session)
    except Exception as exc:
        database = {"connected": False, "error": str(exc)}
    return {
        "status": "ok" if database.get("connected") else "degraded",
        "version": __version__,
        "database": database,
    }
