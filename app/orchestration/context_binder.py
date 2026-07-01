from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.constants import Intent
from app.ner.entity_span_extractor import EntitySpan, SpanExtractionResult
from app.orchestration.intent_registry import (
    EntityScope,
    InheritanceRule,
    capability_for,
    entity_type_for_intent,
)
from app.orchestration.query_features import QueryFeatures
from app.orchestration.schemas import (
    BindingDecision,
    BindingSource,
    BoundTask,
    PlannedTask,
    ReferenceMode,
    TaskPlan,
)
from app.retrieval.normalization import normalize_vietnamese


@dataclass(frozen=True)
class BindingResult:
    decisions: tuple[BindingDecision, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": [
                decision.model_dump(mode="json")
                for decision in self.decisions
            ]
        }


class ContextBinder:
    """Pure task binder.

    Planner tasks are immutable proposals. This component emits binding
    decisions and never mutates planner state.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def bind(
        self,
        *,
        plan: TaskPlan,
        original_query: str,
        history: dict[str, Any],
        span_result: SpanExtractionResult,
        prior_bound_tasks: tuple[BoundTask, ...] = (),
    ) -> BindingResult:
        if not self.settings.enable_context_binder:
            return BindingResult(
                decisions=tuple(
                    self._planner_only_decision(task)
                    for task in sorted(plan.tasks, key=lambda item: item.priority)
                )
            )
        decisions = tuple(
            self.bind_task(
                task=task,
                original_query=original_query,
                history=history,
                span_result=span_result,
                prior_bound_tasks=prior_bound_tasks,
            )
            for task in sorted(plan.tasks, key=lambda item: item.priority)
        )
        return BindingResult(decisions=decisions)

    def bind_task(
        self,
        *,
        task: PlannedTask,
        original_query: str,
        history: dict[str, Any],
        span_result: SpanExtractionResult,
        prior_bound_tasks: tuple[BoundTask, ...] = (),
    ) -> BindingDecision:
        capability = capability_for(task.intent)
        entity_type = _task_entity_type(task)
        planner_entities = tuple(task.planner_entities)
        explicit_spans = _domain_spans(
            span_result.spans,
            entity_type,
            task.planner_query,
        )
        if entity_type is None:
            entity_type = _span_entity_type(explicit_spans)
        explicit_names = tuple(_span_texts(explicit_spans))
        planner_explicit = _explicit_planner_entities(
            planner_entities,
            explicit_spans,
            original_query,
        )
        if not explicit_names and planner_explicit:
            explicit_names = tuple(
                entity
                for entity in planner_entities
                if normalize_vietnamese(entity)
                in normalize_vietnamese(original_query)
            )
        rejected = tuple(
            entity
            for entity in planner_entities
            if normalize_vietnamese(entity)
            not in {
                normalize_vietnamese(name)
                for name in explicit_names
            }
        )

        if capability.entity_scope == EntityScope.NONE:
            return BindingDecision(
                task_id=task.task_id,
                intent=task.intent,
                reference_mode=ReferenceMode.NO_ENTITY,
                binding_source=BindingSource.NONE,
                rejected_planner_entities=planner_entities,
                explicit_spans=tuple(
                    span.as_dict() for span in explicit_spans
                ),
                clarification_required=task.intent == Intent.UNKNOWN,
                clarification_question=(
                    "Bạn vui lòng cho biết cụ thể muốn hỏi về sản phẩm, "
                    "dịch vụ, FAQ hay thông tin phòng khám."
                    if task.intent == Intent.UNKNOWN
                    else None
                ),
                reason_codes=("intent_blocks_entities",),
            )

        implicit = _is_implicit_follow_up(
            original_query,
            task.planner_query,
        )
        if capability.entity_scope == EntityScope.FILTER_ONLY:
            refinement = implicit and _is_filter_refinement(
                task.planner_query
            )
            return BindingDecision(
                task_id=task.task_id,
                intent=task.intent,
                entity_type=entity_type,
                reference_mode=(
                    ReferenceMode.FILTER_REFINEMENT
                    if refinement
                    else ReferenceMode.NO_ENTITY
                ),
                binding_source=(
                    BindingSource.CONVERSATION_STATE
                    if refinement
                    else BindingSource.TASK_FILTERS
                ),
                rejected_planner_entities=planner_entities,
                explicit_spans=tuple(
                    span.as_dict() for span in explicit_spans
                ),
                reason_codes=(
                    "filter_refinement"
                    if refinement
                    else "filter_only_intent",
                ),
            )

        inherited = _inheritable_task(
            prior_bound_tasks,
            entity_type=entity_type,
        )

        if explicit_names:
            return BindingDecision(
                task_id=task.task_id,
                intent=task.intent,
                entity_type=entity_type,
                reference_mode=(
                    ReferenceMode.COMPARE
                    if capability.entity_scope == EntityScope.TWO_OR_MORE
                    else ReferenceMode.EXPLICIT
                ),
                binding_source=BindingSource.EXPLICIT_SPAN,
                entity_names=explicit_names,
                rejected_planner_entities=rejected,
                explicit_spans=tuple(
                    span.as_dict() for span in explicit_spans
                ),
                reason_codes=("explicit_span_authority",),
            )

        if (
            capability.inheritance_rule
            == InheritanceRule.INHERIT_IF_RESOLVED
            and inherited is not None
        ):
            return BindingDecision(
                task_id=task.task_id,
                intent=task.intent,
                entity_type=inherited.entity_type,
                reference_mode=ReferenceMode.IMPLICIT,
                binding_source=BindingSource.SAME_TURN_TASK,
                entity_names=inherited.entity_names,
                inherited_resolved_ids=inherited.resolved_ids,
                inherited_from_task_id=inherited.task_id,
                rejected_planner_entities=planner_entities,
                explicit_spans=tuple(
                    span.as_dict() for span in explicit_spans
                ),
                reason_codes=("inherit_from_resolved_same_turn_task",),
            )

        if capability.entity_scope == EntityScope.OPTIONAL:
            return BindingDecision(
                task_id=task.task_id,
                intent=task.intent,
                entity_type=entity_type,
                reference_mode=ReferenceMode.NO_ENTITY,
                binding_source=BindingSource.NONE,
                rejected_planner_entities=planner_entities,
                explicit_spans=tuple(
                    span.as_dict() for span in explicit_spans
                ),
                reason_codes=("optional_entity_absent",),
            )

        return BindingDecision(
            task_id=task.task_id,
            intent=task.intent,
            entity_type=entity_type,
            reference_mode=(
                ReferenceMode.COMPARE
                if capability.entity_scope == EntityScope.TWO_OR_MORE
                else ReferenceMode.IMPLICIT
                if implicit
                else ReferenceMode.NO_ENTITY
            ),
            binding_source=BindingSource.NONE,
            rejected_planner_entities=planner_entities,
            explicit_spans=tuple(
                span.as_dict() for span in explicit_spans
            ),
            clarification_required=True,
            clarification_question=_clarification_for_scope(
                capability.entity_scope,
                entity_type,
            ),
            reason_codes=("missing_authoritative_entity_context",),
        )

    @staticmethod
    def _planner_only_decision(task: PlannedTask) -> BindingDecision:
        entity_type = _task_entity_type(task)
        return BindingDecision(
            task_id=task.task_id,
            intent=task.intent,
            entity_type=entity_type,
            reference_mode=(
                ReferenceMode.EXPLICIT
                if task.planner_entities
                else ReferenceMode.NO_ENTITY
            ),
            binding_source=(
                BindingSource.PLANNER
                if task.planner_entities
                else BindingSource.NONE
            ),
            entity_names=task.planner_entities,
            reason_codes=("context_binder_disabled",),
        )


def _task_entity_type(task: PlannedTask) -> str | None:
    return entity_type_for_intent(task.intent, task.planner_entity_type)


def _inheritable_task(
    tasks: tuple[BoundTask, ...],
    *,
    entity_type: str | None,
) -> BoundTask | None:
    for task in reversed(tasks):
        if (
            task.resolution_status == "resolved"
            and not task.clarification_required
            and task.resolved_ids
            and task.entity_names
            and (
                entity_type is None
                or task.entity_type == entity_type
            )
        ):
            return task
    return None


def _domain_spans(
    spans: list[EntitySpan],
    entity_type: str | None,
    task_query: str,
) -> list[EntitySpan]:
    if entity_type == "product":
        labels = {"product_name", "brand"}
    elif entity_type == "service":
        labels = {"service_name"}
    else:
        labels = {"product_name", "service_name", "brand"}
    normalized_task = normalize_vietnamese(task_query)
    return [
        span
        for span in spans
        if span.label in labels
        and _span_is_in_task(span, normalized_task)
    ]


def _span_is_in_task(span: EntitySpan, normalized_task: str) -> bool:
    normalized_span = normalize_vietnamese(span.text)
    if normalized_span and normalized_span in normalized_task:
        return True
    catalog_name = normalize_vietnamese(
        str(span.metadata.get("catalog_name") or "")
    )
    return bool(catalog_name and catalog_name in normalized_task)


def _span_texts(spans: list[EntitySpan]) -> list[str]:
    catalog_names = [
        str(span.metadata["catalog_name"])
        for span in spans
        if span.metadata.get("catalog_name")
    ]
    if catalog_names:
        return _dedupe(catalog_names)
    return _dedupe(span.text for span in spans)


def _span_entity_type(spans: list[EntitySpan]) -> str | None:
    if any(
        span.label in {"product_name", "brand"}
        for span in spans
    ):
        return "product"
    if any(span.label == "service_name" for span in spans):
        return "service"
    return None


def _explicit_planner_entities(
    entities: tuple[str, ...],
    explicit_spans: list[EntitySpan],
    original_query: str,
) -> bool:
    if not entities:
        return False
    normalized_query = normalize_vietnamese(original_query)
    span_values = {
        normalize_vietnamese(
            str(span.metadata.get("catalog_name") or span.text)
        )
        for span in explicit_spans
    }
    return any(
        normalize_vietnamese(entity) in normalized_query
        or normalize_vietnamese(entity) in span_values
        for entity in entities
        if normalize_vietnamese(entity)
    )


def _is_implicit_follow_up(
    original_query: str,
    task_query: str,
) -> bool:
    features = QueryFeatures.extract(original_query)
    normalized_original = normalize_vietnamese(original_query).lower().strip()
    normalized_task = normalize_vietnamese(task_query).lower().strip()
    if features.has_implicit_reference:
        return True
    return bool(
        normalized_original
        and normalized_original in normalized_task
        and normalized_original != normalized_task
        and features.has_implicit_reference
    )


def _is_filter_refinement(query: str) -> bool:
    return QueryFeatures.extract(query).is_filter_refinement


def _clarification_for_scope(
    scope: EntityScope,
    entity_type: str | None,
) -> str:
    label = "sản phẩm" if entity_type == "product" else "dịch vụ"
    if scope == EntityScope.TWO_OR_MORE:
        return f"Bạn vui lòng nêu rõ ít nhất hai {label} cần so sánh."
    return f"Bạn đang hỏi {label} nào? Vui lòng cho tôi tên cụ thể."


def _dedupe(values) -> list[str]:
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    )
