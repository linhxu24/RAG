from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.constants import Intent
from app.retrieval.types import RetrievalResult

TrustLevel = Literal["authoritative", "curated", "retrieved"]


def _string_values(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_string_values(item))
        return values
    text = str(value).strip()
    return [text] if text else []


class ReferenceMode(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    MIXED = "mixed"
    COMPARE = "compare"
    FILTER_REFINEMENT = "filter_refinement"
    NO_ENTITY = "no_entity"


class BindingSource(StrEnum):
    EXPLICIT_SPAN = "explicit_span"
    SAME_TURN_TASK = "same_turn_task"
    CONVERSATION_STATE = "conversation_state"
    MIXED_CONTEXT = "mixed_context"
    PLANNER = "planner"
    TASK_FILTERS = "task_filters"
    NONE = "none"


class GateStatus(StrEnum):
    PASS = "pass"
    CLARIFY = "clarify"
    DEGRADED = "degraded"
    BLOCK = "block"


class PlannerSelection(BaseModel):
    """Untrusted selection proposal emitted by the planner."""

    model_config = ConfigDict(frozen=True)

    mode: str = "auto"
    entity_type: str | None = None
    mentions: tuple[str, ...] = ()
    filters: dict[str, Any] = Field(default_factory=dict)
    sort: dict[str, Any] | None = None
    limit: int | None = None

    @field_validator("mentions", mode="before")
    @classmethod
    def normalize_mentions(cls, value: object) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_string_values(value)))

    @field_validator("filters", mode="before")
    @classmethod
    def normalize_filters(cls, value: object) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @field_validator("sort", mode="before")
    @classmethod
    def normalize_sort(cls, value: object) -> dict[str, Any] | None:
        return dict(value) if isinstance(value, dict) else None

    @field_validator("limit", mode="before")
    @classmethod
    def normalize_limit(cls, value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return max(1, min(parsed, 500))


# Transitional import alias. PlannedTask stores this only as an untrusted
# planner proposal; authoritative IDs live exclusively on BoundTask.
TaskSelection = PlannerSelection


class PlannedTask(BaseModel):
    """Read-only, untrusted task proposal from PlannerLLM."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    task_id: str
    intent: Intent
    planner_query: str = Field(
        validation_alias=AliasChoices("planner_query", "query")
    )
    planner_entities: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("planner_entities", "entities"),
    )
    planner_filters: dict[str, Any] = Field(default_factory=dict)
    planner_sort: dict[str, Any] | None = None
    planner_limit: int | None = None
    planner_entity_type: str | None = None
    priority: int = 1
    planner_needs_clarification: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "planner_needs_clarification",
            "needs_clarification",
        ),
    )
    planner_clarification_question: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "planner_clarification_question",
            "clarification_question",
        ),
    )
    @field_validator("planner_query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("planner_query cannot be empty")
        return text

    @field_validator("planner_entities", mode="before")
    @classmethod
    def normalize_string_tuple(cls, value: object) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_string_values(value)))

    @model_validator(mode="before")
    @classmethod
    def flatten_planner_selection(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        selection = payload.get("planner_selection") or payload.get("selection")
        if isinstance(selection, PlannerSelection):
            selection = selection.model_dump(mode="python")
        if isinstance(selection, dict):
            if not payload.get("planner_entities"):
                payload["planner_entities"] = (
                    payload.get("entities")
                    or selection.get("mentions")
                    or ()
                )
            payload.setdefault(
                "planner_filters",
                selection.get("filters") or {},
            )
            payload.setdefault("planner_sort", selection.get("sort"))
            payload.setdefault("planner_limit", selection.get("limit"))
            payload.setdefault(
                "planner_entity_type",
                selection.get("entity_type"),
            )
        return payload

    @field_validator("planner_filters", mode="before")
    @classmethod
    def normalize_planner_filters(cls, value: object) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @field_validator("planner_sort", mode="before")
    @classmethod
    def normalize_planner_sort(
        cls,
        value: object,
    ) -> dict[str, Any] | None:
        return dict(value) if isinstance(value, dict) else None

    @field_validator("planner_limit", mode="before")
    @classmethod
    def normalize_planner_limit(cls, value: object) -> int | None:
        return PlannerSelection.normalize_limit(value)


class TaskPlan(BaseModel):
    """Planner output. This object is diagnostic and never executable."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    tasks: tuple[PlannedTask, ...]
    planner_global_entities: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "planner_global_entities",
            "global_entities",
        ),
    )
    clarification_question: str | None = None
    source: str = "planner"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tasks", mode="before")
    @classmethod
    def normalize_tasks(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return ()

    @field_validator("planner_global_entities", mode="before")
    @classmethod
    def normalize_global_entities(cls, value: object) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_string_values(value)))

    @property
    def primary_intent(self) -> Intent:
        if not self.tasks:
            return Intent.UNKNOWN
        return sorted(self.tasks, key=lambda task: task.priority)[0].intent

    @property
    def confidence(self) -> float:
        if self.source in {"ollama", "openai"}:
            return 0.9
        if self.source == "safe_fallback":
            return 0.25
        return 0.75

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BindingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    intent: Intent
    reference_mode: ReferenceMode
    binding_source: BindingSource
    entity_type: str | None = None
    entity_names: tuple[str, ...] = ()
    inherited_resolved_ids: tuple[str, ...] = ()
    inherited_from_task_id: str | None = None
    rejected_planner_entities: tuple[str, ...] = ()
    explicit_spans: tuple[dict[str, Any], ...] = ()
    clarification_required: bool = False
    clarification_question: str | None = None
    reason_codes: tuple[str, ...] = ()


class TaskResolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    status: str
    entity_type: str | None = None
    entity_names: tuple[str, ...] = ()
    resolved_ids: tuple[str, ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()
    ambiguous_candidates: tuple[dict[str, Any], ...] = ()
    source: str = "database"


class CanonicalFilters(BaseModel):
    """Typed, immutable filter state used by executable tasks."""

    model_config = ConfigDict(frozen=True)

    category_codes: tuple[str, ...] = ()
    category_terms: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    product_names: tuple[str, ...] = ()
    service_ids: tuple[str, ...] = ()
    service_names: tuple[str, ...] = ()
    brand_terms: tuple[str, ...] = ()
    feature_terms: tuple[str, ...] = ()
    symptom_terms: tuple[str, ...] = ()
    price_min: float | None = None
    price_max: float | None = None
    quantity_min: int | None = None
    quantity_max: int | None = None
    duration_min: int | None = None
    duration_max: int | None = None
    stock: str | None = None
    sort_field: str | None = None
    sort_direction: str | None = None
    limit: int | None = None

    @field_validator(
        "category_codes",
        "category_terms",
        "product_ids",
        "product_names",
        "service_ids",
        "service_names",
        "brand_terms",
        "feature_terms",
        "symptom_terms",
        mode="before",
    )
    @classmethod
    def normalize_tuple(cls, value: object) -> tuple[str, ...]:
        return tuple(dict.fromkeys(_string_values(value)))

    def as_constraints(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude_none=True)
        sort_field = data.pop("sort_field", None)
        sort_direction = data.pop("sort_direction", None)
        data.pop("limit", None)
        data["sort"] = (
            {"field": sort_field, "direction": sort_direction}
            if sort_field or sort_direction
            else None
        )
        return data


class BoundTask(BaseModel):
    """Trusted executable task created atomically after binding and resolution."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    intent: Intent
    priority: int = 1
    planner_query: str
    effective_query: str
    entity_type: str | None = None
    entity_names: tuple[str, ...] = ()
    resolved_ids: tuple[str, ...] = ()
    filters: CanonicalFilters = Field(default_factory=CanonicalFilters)
    binding_source: BindingSource = BindingSource.NONE
    reference_mode: ReferenceMode = ReferenceMode.NO_ENTITY
    inherited_from_task_id: str | None = None
    resolution_status: str = "not_applicable"
    clarification_required: bool = False
    clarification_question: str | None = None
    operation: str | None = None
    capability_version: str = "v1"

    @property
    def needs_clarification(self) -> bool:
        return self.clarification_required


class BoundTaskPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    tasks: tuple[BoundTask, ...]
    clarification_question: str | None = None
    source: str = "canonicalizer"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_intent(self) -> Intent:
        if not self.tasks:
            return Intent.UNKNOWN
        return sorted(self.tasks, key=lambda task: task.priority)[0].intent

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ConsistencyViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    task_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ConsistencyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: GateStatus = GateStatus.PASS
    valid_task_ids: tuple[str, ...] = ()
    violations: tuple[ConsistencyViolation, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS


class EvidenceItem(BaseModel):
    task_id: str
    source_type: str
    source_id: str
    text: str
    score: float = 0.0
    trust_level: TrustLevel = "retrieved"
    raw_json: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)
    canonical_key: str | None = None
    asset_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_retrieval(
        cls,
        *,
        task_id: str,
        result: RetrievalResult,
        trust_level: TrustLevel,
    ) -> EvidenceItem:
        asset_id = result.raw_json.get("asset_id") if result.raw_json else None
        return cls(
            task_id=task_id,
            source_type=result.source_type,
            source_id=result.source_id,
            text=result.text,
            score=result.score,
            trust_level=trust_level,
            raw_json=result.raw_json,
            source=result.source,
            canonical_key=result.canonical_key,
            asset_ids=[str(asset_id)] if asset_id else [],
        )

    def to_retrieval_result(self) -> RetrievalResult:
        return RetrievalResult(
            source_type=self.source_type,
            source_id=self.source_id,
            text=self.text,
            score=self.score,
            raw_json=self.raw_json,
            source={**self.source, "task_id": self.task_id, "trust_level": self.trust_level},
            canonical_key=self.canonical_key,
        )


class EvidencePack(BaseModel):
    query: str
    tasks: tuple[BoundTask, ...]
    items: list[EvidenceItem]
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)

    def to_context(self) -> dict[str, Any]:
        results = [item.to_retrieval_result() for item in self.items]
        context_items = [
            {
                "source_type": result.source_type,
                "source_id": result.source_id,
                "text": result.text,
                "raw_json": result.raw_json,
                "source": result.source,
                "score": result.score,
                "canonical_key": result.canonical_key,
            }
            for result in results
        ]
        return {
            "items": context_items,
            "total_chars": sum(len(item["text"]) for item in context_items),
            "tasks": [
                task.model_dump(mode="json")
                for task in self.tasks
                if not task.clarification_required
                and task.intent
                not in {Intent.GREETING, Intent.CHITCHAT, Intent.UNKNOWN}
            ],
            "conflicts": self.conflicts,
            "missing_info": self.missing_info,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_prompt_payload(self) -> dict[str, Any]:
        """Build the synthesis firewall payload.

        Planner proposals, rejected entities, candidates and global planner
        metadata intentionally never cross this boundary.
        """
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "intent": task.intent.value,
                    "effective_query": task.effective_query,
                    "entity_names": list(task.entity_names),
                    "reference_mode": task.reference_mode.value,
                }
                for task in self.tasks
                if not task.clarification_required
            ],
            "evidence": [
                {
                    "task_id": item.task_id,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "trust_level": item.trust_level,
                    "text": item.text,
                    "data": _compact_data(item.source_type, item.raw_json),
                }
                for item in self.items
            ],
            "conflicts": self.conflicts,
            "missing_info": self.missing_info,
        }


def _compact_data(source_type: str, raw: dict[str, Any]) -> dict[str, Any]:
    fields_by_source = {
        "product": (
            "name",
            "category",
            "brand",
            "model",
            "description",
            "price",
            "currency",
            "quantity",
            "link",
        ),
        "service": (
            "name",
            "category_code",
            "source_category",
            "description",
            "duration_minutes",
            "price",
            "currency",
            "symptoms",
            "indications",
            "contraindications",
        ),
        "faq": ("question", "answer", "category"),
        "clinic_info": ("key", "value"),
    }
    fields = fields_by_source.get(source_type)
    if fields is None:
        return {}
    return {field: raw[field] for field in fields if raw.get(field) is not None}
