from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db
from app.generation.schemas import ChatRequest, ChatResponse
from app.services.chat import ChatService

router = APIRouter(tags=["chat"])


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(get_settings())


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session: Session = Depends(get_db)) -> ChatResponse:
    try:
        return await get_chat_service().chat(session, request)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
