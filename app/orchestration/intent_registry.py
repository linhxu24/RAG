from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.constants import Intent
from app.orchestration.schemas import ReferenceMode


class EntityScope(StrEnum):
    NONE = "none"
    EXACTLY_ONE = "exactly_one"
    TWO_OR_MORE = "two_or_more"
    OPTIONAL = "optional"
    FILTER_ONLY = "filter_only"


class InheritanceRule(StrEnum):
    NO_INHERIT = "no_inherit"
    INHERIT_IF_RESOLVED = "inherit_if_resolved"
    EXPLICIT_OR_MEMORY = "explicit_or_memory"
    MEMORY_PLUS_EXPLICIT = "memory_plus_explicit"
    EXPLICIT_MULTI = "explicit_multi"


class ClarificationPolicy(StrEnum):
    NEVER = "never"
    ALWAYS = "always"
    IF_AMBIGUOUS = "if_ambiguous"
    IF_ANY_MISSING = "if_any_missing"
    IF_MISSING_KEY = "if_missing_key"
    IF_NO_EVIDENCE = "if_no_evidence"
    IF_AMBIGUOUS_FILTER = "if_ambiguous_filter"


class EvidenceContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed_source_types: tuple[str, ...] = ()
    authoritative_required: bool = False
    match_resolved_ids: bool = False
    minimum_items: int = 0


class SuggestionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed_next_intents: tuple[Intent, ...] = ()
    topic_sequence: tuple[str, ...] = ()
    entity_required: bool = False
    evidence_required: bool = False
    max_suggestions: int = 3


class IntentCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent: Intent
    entity_scope: EntityScope
    allowed_entity_types: tuple[str, ...] = ()
    inheritance_rule: InheritanceRule
    allowed_reference_modes: tuple[ReferenceMode, ...]
    allowed_tools: tuple[str, ...] = ()
    clarification_policy: ClarificationPolicy
    evidence_contract: EvidenceContract = EvidenceContract()
    persist_to_memory: bool = False
    suggestion_policy: SuggestionPolicy = SuggestionPolicy()


_NO_ENTITY_MODES = (ReferenceMode.NO_ENTITY,)
_DETAIL_MODES = (
    ReferenceMode.EXPLICIT,
    ReferenceMode.IMPLICIT,
)


INTENT_CAPABILITIES: dict[Intent, IntentCapability] = {
    Intent.GREETING: IntentCapability(
        intent=Intent.GREETING,
        entity_scope=EntityScope.NONE,
        inheritance_rule=InheritanceRule.NO_INHERIT,
        allowed_reference_modes=_NO_ENTITY_MODES,
        clarification_policy=ClarificationPolicy.NEVER,
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.PRODUCT_LIST,
                Intent.SERVICE_LIST,
                Intent.CLINIC_INFO,
            ),
            topic_sequence=("product_catalog", "service_catalog", "clinic_location"),
        ),
    ),
    Intent.CHITCHAT: IntentCapability(
        intent=Intent.CHITCHAT,
        entity_scope=EntityScope.NONE,
        inheritance_rule=InheritanceRule.NO_INHERIT,
        allowed_reference_modes=_NO_ENTITY_MODES,
        clarification_policy=ClarificationPolicy.NEVER,
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.PRODUCT_LIST,
                Intent.SERVICE_LIST,
                Intent.CLINIC_INFO,
            ),
            topic_sequence=("product_catalog", "service_catalog", "clinic_location"),
        ),
    ),
    Intent.CLINIC_INFO: IntentCapability(
        intent=Intent.CLINIC_INFO,
        entity_scope=EntityScope.NONE,
        inheritance_rule=InheritanceRule.NO_INHERIT,
        allowed_reference_modes=_NO_ENTITY_MODES,
        allowed_tools=("clinic_info_tool",),
        clarification_policy=ClarificationPolicy.IF_MISSING_KEY,
        evidence_contract=EvidenceContract(
            allowed_source_types=("clinic_info",),
            authoritative_required=True,
            minimum_items=1,
        ),
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.CLINIC_INFO,
                Intent.PRODUCT_LIST,
                Intent.SERVICE_LIST,
            ),
            topic_sequence=("clinic_hours", "clinic_contact", "clinic_location"),
            evidence_required=True,
        ),
    ),
    Intent.FAQ: IntentCapability(
        intent=Intent.FAQ,
        entity_scope=EntityScope.OPTIONAL,
        allowed_entity_types=("product", "service"),
        inheritance_rule=InheritanceRule.INHERIT_IF_RESOLVED,
        allowed_reference_modes=(
            ReferenceMode.NO_ENTITY,
            ReferenceMode.EXPLICIT,
            ReferenceMode.IMPLICIT,
            ReferenceMode.MIXED,
        ),
        allowed_tools=("faq_tool", "document_rag_tool"),
        clarification_policy=ClarificationPolicy.IF_NO_EVIDENCE,
        evidence_contract=EvidenceContract(
            allowed_source_types=("faq", "chunk", "table_row"),
            minimum_items=1,
        ),
        persist_to_memory=False,
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.FAQ,
                Intent.PRODUCT_DETAIL,
                Intent.SERVICE_DETAIL,
                Intent.CLINIC_INFO,
            ),
            topic_sequence=(
                "price",
                "duration",
                "availability",
                "pain",
                "safety",
                "clinic_hours",
            ),
            evidence_required=True,
        ),
    ),
    Intent.PRODUCT_LIST: IntentCapability(
        intent=Intent.PRODUCT_LIST,
        entity_scope=EntityScope.FILTER_ONLY,
        allowed_entity_types=("product",),
        inheritance_rule=InheritanceRule.NO_INHERIT,
        allowed_reference_modes=(
            ReferenceMode.NO_ENTITY,
            ReferenceMode.FILTER_REFINEMENT,
        ),
        allowed_tools=("product_tool",),
        clarification_policy=ClarificationPolicy.IF_AMBIGUOUS_FILTER,
        evidence_contract=EvidenceContract(
            allowed_source_types=("product",),
            authoritative_required=True,
        ),
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.PRODUCT_LIST,
                Intent.PRODUCT_DETAIL,
            ),
            topic_sequence=("availability", "price_sort", "budget", "detail"),
            evidence_required=True,
        ),
    ),
    Intent.PRODUCT_DETAIL: IntentCapability(
        intent=Intent.PRODUCT_DETAIL,
        entity_scope=EntityScope.EXACTLY_ONE,
        allowed_entity_types=("product",),
        inheritance_rule=InheritanceRule.EXPLICIT_OR_MEMORY,
        allowed_reference_modes=_DETAIL_MODES,
        allowed_tools=("product_tool",),
        clarification_policy=ClarificationPolicy.IF_AMBIGUOUS,
        evidence_contract=EvidenceContract(
            allowed_source_types=("product",),
            authoritative_required=True,
            match_resolved_ids=True,
            minimum_items=1,
        ),
        persist_to_memory=True,
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.PRODUCT_DETAIL,
                Intent.FAQ,
                Intent.CLINIC_INFO,
            ),
            topic_sequence=("price", "availability", "usage", "safety", "clinic_hours"),
            entity_required=True,
            evidence_required=True,
        ),
    ),
    Intent.PRODUCT_COMPARE: IntentCapability(
        intent=Intent.PRODUCT_COMPARE,
        entity_scope=EntityScope.TWO_OR_MORE,
        allowed_entity_types=("product",),
        inheritance_rule=InheritanceRule.MEMORY_PLUS_EXPLICIT,
        allowed_reference_modes=(
            ReferenceMode.COMPARE,
            ReferenceMode.MIXED,
        ),
        allowed_tools=("product_tool",),
        clarification_policy=ClarificationPolicy.IF_ANY_MISSING,
        evidence_contract=EvidenceContract(
            allowed_source_types=("product",),
            authoritative_required=True,
            match_resolved_ids=True,
            minimum_items=2,
        ),
        persist_to_memory=True,
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.PRODUCT_DETAIL,
                Intent.PRODUCT_LIST,
            ),
            topic_sequence=("availability", "price", "detail"),
            evidence_required=True,
        ),
    ),
    Intent.SERVICE_LIST: IntentCapability(
        intent=Intent.SERVICE_LIST,
        entity_scope=EntityScope.FILTER_ONLY,
        allowed_entity_types=("service",),
        inheritance_rule=InheritanceRule.NO_INHERIT,
        allowed_reference_modes=(
            ReferenceMode.NO_ENTITY,
            ReferenceMode.FILTER_REFINEMENT,
        ),
        allowed_tools=("service_tool",),
        clarification_policy=ClarificationPolicy.IF_AMBIGUOUS_FILTER,
        evidence_contract=EvidenceContract(
            allowed_source_types=("service",),
            authoritative_required=True,
        ),
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.SERVICE_LIST,
                Intent.SERVICE_DETAIL,
                Intent.CLINIC_INFO,
            ),
            topic_sequence=("price_sort", "duration_sort", "budget", "detail"),
            evidence_required=True,
        ),
    ),
    Intent.SERVICE_DETAIL: IntentCapability(
        intent=Intent.SERVICE_DETAIL,
        entity_scope=EntityScope.EXACTLY_ONE,
        allowed_entity_types=("service",),
        inheritance_rule=InheritanceRule.EXPLICIT_OR_MEMORY,
        allowed_reference_modes=_DETAIL_MODES,
        allowed_tools=("service_tool",),
        clarification_policy=ClarificationPolicy.IF_AMBIGUOUS,
        evidence_contract=EvidenceContract(
            allowed_source_types=("service",),
            authoritative_required=True,
            match_resolved_ids=True,
            minimum_items=1,
        ),
        persist_to_memory=True,
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.SERVICE_DETAIL,
                Intent.FAQ,
                Intent.CLINIC_INFO,
            ),
            topic_sequence=(
                "price",
                "duration",
                "pain",
                "safety",
                "clinic_hours",
            ),
            entity_required=True,
            evidence_required=True,
        ),
    ),
    Intent.UNKNOWN: IntentCapability(
        intent=Intent.UNKNOWN,
        entity_scope=EntityScope.NONE,
        inheritance_rule=InheritanceRule.NO_INHERIT,
        allowed_reference_modes=_NO_ENTITY_MODES,
        clarification_policy=ClarificationPolicy.ALWAYS,
        suggestion_policy=SuggestionPolicy(
            allowed_next_intents=(
                Intent.PRODUCT_LIST,
                Intent.SERVICE_LIST,
                Intent.CLINIC_INFO,
            ),
            topic_sequence=("product_catalog", "service_catalog", "clinic_location"),
        ),
    ),
}


def capability_for(intent: Intent) -> IntentCapability:
    try:
        return INTENT_CAPABILITIES[intent]
    except KeyError as exc:  # pragma: no cover - guarded by startup validation
        raise ValueError(f"Intent is not registered: {intent}") from exc


def validate_intent_registry() -> None:
    missing = set(Intent) - set(INTENT_CAPABILITIES)
    extra = set(INTENT_CAPABILITIES) - set(Intent)
    if missing or extra:
        raise RuntimeError(
            "IntentCapabilityRegistry coverage mismatch: "
            f"missing={sorted(item.value for item in missing)}, "
            f"extra={sorted(item.value for item in extra)}"
        )
    known_tools = {
        "product_tool",
        "service_tool",
        "clinic_info_tool",
        "faq_tool",
        "document_rag_tool",
    }
    for intent, capability in INTENT_CAPABILITIES.items():
        unknown_tools = set(capability.allowed_tools) - known_tools
        if unknown_tools:
            raise RuntimeError(
                f"{intent.value} registers unknown tools: {sorted(unknown_tools)}"
            )
        if (
            capability.entity_scope == EntityScope.NONE
            and capability.allowed_entity_types
        ):
            raise RuntimeError(
                f"{intent.value} has entity_scope=none but allows entity types"
            )
        if (
            capability.entity_scope == EntityScope.TWO_OR_MORE
            and capability.evidence_contract.minimum_items < 2
        ):
            raise RuntimeError(
                f"{intent.value} compare contract must require at least two evidence items"
            )
        unknown_next_intents = (
            set(capability.suggestion_policy.allowed_next_intents) - set(Intent)
        )
        if unknown_next_intents:
            raise RuntimeError(
                f"{intent.value} registers unknown suggestion intents: "
                f"{sorted(str(item) for item in unknown_next_intents)}"
            )
