from __future__ import annotations

import re
from typing import Any

from app.constants import Intent
from app.orchestration.intent_registry import EntityScope, capability_for
from app.orchestration.schemas import (
    BindingDecision,
    BoundTask,
    CanonicalFilters,
    PlannedTask,
    ReferenceMode,
    TaskResolution,
)
from app.retrieval.router import IntentRouter


class TaskCanonicalizer:
    """Atomically build trusted executable tasks."""

    def canonicalize(
        self,
        *,
        task: PlannedTask,
        decision: BindingDecision,
        resolution: TaskResolution,
        history: dict[str, Any],
    ) -> BoundTask:
        capability = capability_for(task.intent)
        entity_names = (
            resolution.entity_names
            if resolution.entity_names
            else decision.entity_names
        )
        resolved_ids = resolution.resolved_ids
        filters = self._canonical_filters(
            task=task,
            decision=decision,
            entity_names=entity_names,
            resolved_ids=resolved_ids,
            history=history,
        )
        clarification_required = (
            decision.clarification_required
            or resolution.status in {"ambiguous", "not_found", "partial", "missing_context"}
            and capability.entity_scope
            in {EntityScope.EXACTLY_ONE, EntityScope.TWO_OR_MORE}
        )
        clarification_question = (
            decision.clarification_question
            or self._resolution_clarification(
                task.intent,
                resolution.status,
                resolution.ambiguous_candidates,
            )
            if clarification_required
            else None
        )
        effective_query = self._effective_query(
            task=task,
            decision=decision,
            entity_names=entity_names,
        )
        return BoundTask(
            task_id=task.task_id,
            intent=task.intent,
            priority=task.priority,
            planner_query=task.planner_query,
            effective_query=effective_query,
            entity_type=resolution.entity_type or decision.entity_type,
            entity_names=entity_names,
            resolved_ids=resolved_ids,
            filters=filters,
            binding_source=decision.binding_source,
            reference_mode=decision.reference_mode,
            inherited_from_task_id=decision.inherited_from_task_id,
            resolution_status=resolution.status,
            clarification_required=clarification_required,
            clarification_question=clarification_question,
            operation=_operation(task.intent, effective_query),
        )

    @staticmethod
    def _canonical_filters(
        *,
        task: PlannedTask,
        decision: BindingDecision,
        entity_names: tuple[str, ...],
        resolved_ids: tuple[str, ...],
        history: dict[str, Any],
    ) -> CanonicalFilters:
        values = dict(task.planner_filters)
        if decision.reference_mode == ReferenceMode.FILTER_REFINEMENT:
            state = history.get("state") if isinstance(history, dict) else None
            last_filters = (
                state.get("last_filters")
                if isinstance(state, dict)
                and isinstance(state.get("last_filters"), dict)
                else {}
            )
            domain = (
                "product"
                if task.intent == Intent.PRODUCT_LIST
                else "service"
            )
            previous = last_filters.get(domain)
            if isinstance(previous, dict):
                values = {**previous, **values}

        for key in ("entities", "names", "name_terms"):
            values.pop(key, None)
        if task.intent in {Intent.PRODUCT_DETAIL, Intent.PRODUCT_COMPARE}:
            values.pop("product_names", None)
            values.pop("product_ids", None)
            values["product_names"] = list(entity_names)
            values["product_ids"] = list(resolved_ids)
        elif task.intent == Intent.SERVICE_DETAIL:
            values.pop("service_names", None)
            values.pop("service_ids", None)
            values["service_names"] = list(entity_names)
            values["service_ids"] = list(resolved_ids)

        sort = task.planner_sort or {}
        return CanonicalFilters(
            category_codes=values.get("category_codes", ()),
            category_terms=values.get("category_terms", ()),
            product_ids=values.get("product_ids", ()),
            product_names=values.get("product_names", ()),
            service_ids=values.get("service_ids", ()),
            service_names=values.get("service_names", ()),
            brand_terms=values.get("brand_terms")
            or values.get("brands")
            or (),
            feature_terms=values.get("feature_terms", ()),
            symptom_terms=values.get("symptom_terms", ()),
            price_min=values.get("price_min"),
            price_max=values.get("price_max"),
            quantity_min=values.get("quantity_min"),
            quantity_max=values.get("quantity_max"),
            duration_min=values.get("duration_min"),
            duration_max=values.get("duration_max"),
            stock=values.get("stock"),
            sort_field=sort.get("field") or values.get("sort_by"),
            sort_direction=sort.get("direction")
            or values.get("sort_direction"),
            limit=task.planner_limit or values.get("limit"),
        )

    @staticmethod
    def _effective_query(
        *,
        task: PlannedTask,
        decision: BindingDecision,
        entity_names: tuple[str, ...],
    ) -> str:
        query = task.planner_query.strip()
        for rejected in decision.rejected_planner_entities:
            query = re.sub(
                re.escape(rejected),
                "",
                query,
                flags=re.IGNORECASE,
            )
        query = re.sub(r"\s{2,}", " ", query).strip(" ,.;:-")
        if not entity_names:
            return query or task.planner_query.strip()
        normalized_query = IntentRouter._normalize(query)
        missing = [
            name
            for name in entity_names
            if IntentRouter._normalize(name) not in normalized_query
        ]
        if not missing:
            return query
        prefix = " và ".join(entity_names)
        if task.intent == Intent.PRODUCT_COMPARE:
            return f"So sánh {prefix}. {query}"
        return f"{prefix}. {query}"

    @staticmethod
    def _resolution_clarification(
        intent: Intent,
        status: str,
        ambiguous_candidates: tuple[dict[str, Any], ...],
    ) -> str:
        label = "sản phẩm" if intent.name.startswith("PRODUCT_") else "dịch vụ"
        if status == "ambiguous" and ambiguous_candidates:
            names = [
                str(item.get("name"))
                for item in ambiguous_candidates[:5]
                if item.get("name")
            ]
            if names:
                return (
                    f"Tôi thấy nhiều {label} gần giống: "
                    f"{', '.join(names)}. Bạn muốn hỏi mục nào?"
                )
        if intent == Intent.PRODUCT_COMPARE:
            return "Bạn vui lòng nêu rõ ít nhất hai sản phẩm cần so sánh."
        return (
            f"Tôi chưa xác định được chính xác {label}. "
            "Bạn vui lòng nhập tên cụ thể hơn."
        )


def _operation(intent: Intent, query: str) -> str | None:
    normalized = IntentRouter._normalize(query)
    if intent in {Intent.PRODUCT_LIST, Intent.SERVICE_LIST}:
        return "list"
    if intent == Intent.PRODUCT_COMPARE:
        return "compare"
    if intent in {Intent.PRODUCT_DETAIL, Intent.SERVICE_DETAIL}:
        if any(word in normalized for word in ("con hang", "so luong", "ton kho")):
            return "availability_lookup"
        if any(word in normalized for word in ("gia", "chi phi", "bao nhieu")):
            return "price_lookup"
        if any(word in normalized for word in ("mat bao lau", "thoi gian", "bao lau")):
            return "duration_lookup"
        return "detail_lookup"
    if intent == Intent.FAQ:
        return "answer_question"
    if intent == Intent.CLINIC_INFO:
        return "fact_lookup"
    if intent in {Intent.GREETING, Intent.CHITCHAT}:
        return "social"
    return None
