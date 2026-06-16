from typing import Any, Literal

from pydantic import BaseModel, Field

from app.constants import Intent


class EntityReference(BaseModel):
    type: Literal["product", "service", "faq", "clinic_info", "unknown"]
    name: str
    matched_id: str | None = None


class ResultItem(BaseModel):
    type: str
    id: str
    name: str | None = None
    chunk_id: str | None = None
    row_id: str | None = None
    doc_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class SourceReference(BaseModel):
    source_type: str
    source_id: str
    doc_id: str | None = None
    page_number: int | None = None


class ResultBody(BaseModel):
    text: str
    items: list[ResultItem] = Field(default_factory=list)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    missing_assets: list[str] = Field(default_factory=list)


class SafetyInfo(BaseModel):
    medical_disclaimer_required: bool = False
    needs_human_support: bool = False


class GeneratedResponse(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    answer_type: Literal["direct_data", "rag", "clarification", "greeting", "chitchat", "fallback"]
    entities: list[EntityReference] = Field(default_factory=list)
    result: ResultBody
    safety: SafetyInfo = Field(default_factory=SafetyInfo)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)
    session_id: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    debug: bool = False


class ChatResponse(BaseModel):
    trace_id: str
    intent: Intent
    answer_type: str | None = None
    answer: ResultBody
    safety: SafetyInfo = Field(default_factory=SafetyInfo)
    debug: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
