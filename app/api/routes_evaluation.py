import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import __version__
from app.config import get_settings
from app.db.models import EvaluationRun
from app.db.session import get_db
from app.evaluation.datasets import ensure_dataset
from app.evaluation.runner import run_pipeline_evaluation, run_router_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationRequest(BaseModel):
    dataset_path: str | None = None
    dataset_name: str = "dental_basic_eval"
    dataset_version: str = "2.0"
    mode: Literal["router", "e2e", "all"] = "router"
    profile: Literal["deterministic", "production"] = "deterministic"
    data_version: str = "unknown"


@router.post("/run")
async def run_evaluation(
    request: EvaluationRequest,
    session: Session = Depends(get_db),
) -> dict:
    settings = get_settings()
    effective_settings = (
        settings.model_copy(
            update={
                "enable_llm_router": False,
                "enable_hyde": False,
                "enable_reranker": False,
                "ollama_timeout_seconds": min(settings.ollama_timeout_seconds, 15),
            }
        )
        if request.profile == "deterministic"
        else settings
    )
    path = Path(request.dataset_path) if request.dataset_path else settings.eval_dataset_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Evaluation dataset not found: {path}")
    try:
        dataset, cases = ensure_dataset(
            session,
            path=path,
            name=request.dataset_name,
            version=request.dataset_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    run = EvaluationRun(
        dataset_id=dataset.dataset_id,
        pipeline_version=__version__,
        data_version=request.data_version,
        config_snapshot={
            "dataset_content_hash": dataset.content_hash,
            "evaluation_profile": request.profile,
            "embedding_model": effective_settings.embedding_model,
            "embedding_dim": effective_settings.embedding_dim,
            "router_model": effective_settings.ollama_router_model,
            "generation_model": effective_settings.ollama_generation_model,
            "enable_llm_router": effective_settings.enable_llm_router,
            "enable_hyde": effective_settings.enable_hyde,
            "enable_reranker": effective_settings.enable_reranker,
            "ollama_timeout_seconds": effective_settings.ollama_timeout_seconds,
            "dense_top_k": effective_settings.dense_top_k,
            "sparse_top_k": effective_settings.sparse_top_k,
            "final_top_k": effective_settings.final_top_k,
        },
        status="running",
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    try:
        if request.mode == "router":
            metrics = {"router": run_router_evaluation(session, run, cases)}
        else:
            metrics = await run_pipeline_evaluation(
                session,
                effective_settings,
                run,
                cases,
            )
        run.metrics = metrics
        run.status = "completed"
        run.ended_at = datetime.now(UTC)
        session.add(run)
        session.commit()
    except Exception as exc:
        run.status = "failed"
        run.ended_at = datetime.now(UTC)
        run.metrics = {"error": str(exc)}
        session.add(run)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
    return {
        "eval_run_id": str(run.eval_run_id),
        "dataset_id": str(dataset.dataset_id),
        "status": run.status,
        "metrics": run.metrics,
    }


@router.get("/runs/{eval_run_id}")
def get_evaluation_run(
    eval_run_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> dict:
    run = session.get(EvaluationRun, eval_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return {
        "eval_run_id": str(run.eval_run_id),
        "dataset_id": str(run.dataset_id) if run.dataset_id else None,
        "pipeline_version": run.pipeline_version,
        "data_version": run.data_version,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "status": run.status,
        "metrics": run.metrics,
        "config_snapshot": run.config_snapshot,
    }
