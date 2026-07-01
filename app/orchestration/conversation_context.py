from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.constants import Intent
from app.orchestration.intent_registry import domain_for_intent


@dataclass(frozen=True)
class ConversationContext:
    turn_count: int = 0
    last_intent: Intent | None = None
    last_entity_names: tuple[str, ...] = ()
    last_entity_type: str | None = None
    last_compare_entities: tuple[str, ...] = ()
    clarification_pending: bool = False
    topic_stack: tuple[str, ...] = ()
    active_product_ids: tuple[str, ...] = ()
    active_product_names: tuple[str, ...] = ()
    active_service_ids: tuple[str, ...] = ()
    active_service_names: tuple[str, ...] = ()
    last_filters: dict[str, Any] = field(default_factory=dict)
    active_domain_override: str | None = None

    @classmethod
    def from_history(cls, history: dict[str, Any]) -> ConversationContext:
        if not isinstance(history, dict):
            return cls()

        entities: list[str] = []
        last_intent: Intent | None = None
        product_ids: list[str] = []
        product_names: list[str] = []
        service_ids: list[str] = []
        service_names: list[str] = []
        last_filters: dict[str, Any] = {}
        active_domain: str | None = None
        clarification_pending = False
        topic_stack: tuple[str, ...] = ()

        state = history.get("state")
        if isinstance(state, dict):
            product_ids = _string_values(state.get("active_product_ids"))
            product_names = _string_values(state.get("active_product_names"))
            service_ids = _string_values(state.get("active_service_ids"))
            service_names = _string_values(state.get("active_service_names"))
            raw_domain = str(state.get("active_domain") or "").strip()
            active_domain = raw_domain if raw_domain in {"product", "service"} else None
            filters = state.get("last_filters")
            last_filters = filters if isinstance(filters, dict) else {}
            clarification_pending = bool(state.get("clarification_pending"))
            topic_stack = tuple(_string_values(state.get("topic_stack")))
            state_intents = _string_values(state.get("last_intents"))
            if not topic_stack:
                topic_stack = tuple(state_intents)
            for value in state_intents:
                if last_intent is None:
                    try:
                        last_intent = Intent(value)
                    except ValueError:
                        pass

        turns = history.get("turns")
        turn_count = len(turns) if isinstance(turns, list) else 0
        if isinstance(turns, list):
            for turn in reversed(turns):
                if not isinstance(turn, dict):
                    continue
                clarification_pending = clarification_pending or bool(
                    turn.get("clarification_pending")
                )
                if last_intent is None:
                    for value in turn.get("detected_intents") or []:
                        try:
                            last_intent = Intent(value)
                            break
                        except ValueError:
                            continue
                payload = turn.get("entities") or {}
                if isinstance(payload, dict):
                    global_entities = payload.get("global") or []
                    if isinstance(global_entities, list):
                        entities.extend(str(item) for item in global_entities if item)
                    task_entities = payload.get("tasks") or {}
                    if isinstance(task_entities, dict):
                        for values in task_entities.values():
                            if isinstance(values, list):
                                entities.extend(str(item) for item in values if item)
                resolved = turn.get("resolved_ids") or {}
                if isinstance(resolved, dict):
                    product_ids.extend(_resolved_ids(resolved, "product"))
                    service_ids.extend(_resolved_ids(resolved, "service"))
                if entities and last_intent:
                    break

        last_entity_names = tuple(
            dict.fromkeys([*service_names, *product_names, *entities])
        )
        entity_type = (
            active_domain
            or ("service" if service_names or service_ids else None)
            or ("product" if product_names or product_ids else None)
            or (_intent_domain(last_intent) if last_intent else None)
        )
        compare_entities = (
            last_entity_names
            if last_intent == Intent.PRODUCT_COMPARE and len(last_entity_names) >= 2
            else ()
        )
        if not topic_stack and last_intent is not None:
            topic_stack = (last_intent.value,)

        return cls(
            turn_count=turn_count,
            last_intent=last_intent,
            last_entity_names=last_entity_names,
            last_entity_type=entity_type,
            last_compare_entities=compare_entities,
            clarification_pending=clarification_pending,
            topic_stack=topic_stack,
            active_product_ids=tuple(dict.fromkeys(product_ids)),
            active_product_names=tuple(dict.fromkeys(product_names)),
            active_service_ids=tuple(dict.fromkeys(service_ids)),
            active_service_names=tuple(dict.fromkeys(service_names)),
            last_filters=last_filters,
            active_domain_override=active_domain,
        )

    @property
    def has_active_entity(self) -> bool:
        return bool(
            self.last_entity_names
            or self.active_product_ids
            or self.active_service_ids
        )

    @property
    def active_domain(self) -> str | None:
        if self.active_domain_override:
            return self.active_domain_override
        if self.last_entity_type in {"product", "service"}:
            return self.last_entity_type
        if self.active_service_ids or self.active_service_names:
            return "service"
        if self.active_product_ids or self.active_product_names:
            return "product"
        return _intent_domain(self.last_intent) if self.last_intent else None

    @property
    def is_fresh_session(self) -> bool:
        return self.turn_count == 0 and self.last_intent is None and not self.has_active_entity

    def as_legacy_dict(self) -> dict[str, Any]:
        return {
            "entities": list(self.last_entity_names),
            "last_intent": self.last_intent,
            "active_product_ids": list(self.active_product_ids),
            "active_product_names": list(self.active_product_names),
            "active_service_ids": list(self.active_service_ids),
            "active_service_names": list(self.active_service_names),
            "last_filters": self.last_filters,
            "active_domain": self.active_domain,
        }


def build_conversation_context(history: dict[str, Any]) -> ConversationContext:
    return ConversationContext.from_history(history)


def _intent_domain(intent: Intent | None) -> str | None:
    if intent is None:
        return None
    try:
        return domain_for_intent(intent)
    except ValueError:
        return None


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


def _resolved_ids(resolved: dict[str, Any], entity_type: str) -> list[str]:
    values: list[str] = []
    for key in (entity_type, f"{entity_type}s", f"{entity_type}_ids"):
        values.extend(_string_values(resolved.get(key)))
    return values
