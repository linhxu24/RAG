from dataclasses import dataclass

from app.config import Settings
from app.constants import Intent, RetrievalMode
from app.retrieval.entity_resolver import EntityResolution
from app.retrieval.normalization import normalize_vietnamese
from app.retrieval.router import RouterResult
from app.retrieval.types import RetrievalResult


@dataclass
class RetrievalPlan:
    mode: RetrievalMode
    run_structured: bool
    run_dense: bool
    run_sparse: bool
    run_hyde: bool
    use_reranker: bool
    reason: str

    @property
    def uses_hybrid(self) -> bool:
        return self.run_dense or self.run_sparse

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "run_structured": self.run_structured,
            "run_dense": self.run_dense,
            "run_sparse": self.run_sparse,
            "run_hyde": self.run_hyde,
            "use_reranker": self.use_reranker,
            "reason": self.reason,
        }


class RetrievalPlanner:
    TEMPLATE_INTENTS = {Intent.GREETING}
    DIRECT_SQL_INTENTS = {
        Intent.CLINIC_INFO,
        Intent.PRODUCT_LIST,
        Intent.SERVICE_LIST,
    }

    def __init__(self, settings: Settings):
        self.settings = settings

    def plan(
        self,
        *,
        query: str,
        routed: RouterResult,
        entities: EntityResolution,
        structured: list[RetrievalResult],
    ) -> RetrievalPlan:
        intent = routed.intent
        if intent in self.TEMPLATE_INTENTS:
            return self._plan(RetrievalMode.TEMPLATE, False, "template_intent")
        if intent == Intent.CHITCHAT:
            return self._plan(
                RetrievalMode.NO_RAG_LLM,
                False,
                "chitchat_no_rag_llm",
                run_structured=False,
            )
        if (
            routed.needs_clarification
            or routed.confidence < self.settings.confidence_threshold
        ):
            return self._plan(
                RetrievalMode.CLARIFY,
                False,
                "router_confidence_below_threshold",
                run_structured=False,
            )
        if intent in self.DIRECT_SQL_INTENTS:
            return self._plan(RetrievalMode.DIRECT_SQL, False, "direct_sql_intent")
        if intent == Intent.UNKNOWN:
            if self._has_dental_signal(query):
                return self._plan(
                    RetrievalMode.HYBRID,
                    True,
                    "unknown_with_dental_signal",
                    run_structured=False,
                )
            return self._plan(RetrievalMode.CLARIFY, False, "unknown_without_dental_signal")
        if intent == Intent.FAQ:
            if structured and structured[0].score >= self.settings.faq_direct_threshold:
                return self._plan(
                    RetrievalMode.STRUCTURED_ONLY,
                    False,
                    "high_confidence_faq_match",
                )
            return self._plan(
                RetrievalMode.STRUCTURED_THEN_HYBRID,
                True,
                "faq_requires_hybrid_support",
                use_reranker=False,
                run_hyde=False,
            )
        if intent in {Intent.PRODUCT_DETAIL, Intent.SERVICE_DETAIL}:
            if not structured:
                return self._plan(
                    RetrievalMode.CLARIFY,
                    False,
                    "structured_entity_not_found",
                )
            if (
                routed.source == "ollama"
                and len(structured) == 1
                and structured[0].score >= self.settings.structured_direct_threshold
            ):
                return self._plan(
                    RetrievalMode.STRUCTURED_ONLY,
                    False,
                    "router_entity_matches_structured_record",
                )
            if self._has_authoritative_detail_match(entities, structured):
                return self._plan(
                    RetrievalMode.STRUCTURED_ONLY,
                    False,
                    "resolved_entity_matches_structured_record",
                )
            if (
                entities.status in {"ambiguous", "not_found"}
                or entities.best_score < self.settings.structured_direct_threshold
            ):
                return self._plan(
                    RetrievalMode.CLARIFY,
                    False,
                    "fuzzy_or_ambiguous_structured_match",
                )
            return self._plan(
                RetrievalMode.STRUCTURED_ONLY,
                False,
                "high_confidence_structured_match",
            )
        if intent == Intent.PRODUCT_COMPARE:
            if len(structured) >= 2 and (
                entities.status == "resolved" or routed.source == "ollama"
            ):
                return self._plan(
                    RetrievalMode.STRUCTURED_ONLY,
                    False,
                    "all_compare_entities_resolved",
                )
            return self._plan(
                RetrievalMode.CLARIFY,
                False,
                "compare_entities_incomplete",
            )
        return self._plan(RetrievalMode.HYBRID, True, "default_hybrid")

    def _has_authoritative_detail_match(
        self,
        entities: EntityResolution,
        structured: list[RetrievalResult],
    ) -> bool:
        if entities.status != "resolved" or len(entities.selected) != 1 or len(structured) != 1:
            return False
        selected = entities.selected[0]
        result = structured[0]
        return (
            selected.entity_id == result.source_id
            and result.score >= self.settings.structured_direct_threshold
        )

    def _plan(
        self,
        mode: RetrievalMode,
        hybrid: bool,
        reason: str,
        *,
        run_structured: bool = True,
        use_reranker: bool | None = None,
        run_hyde: bool | None = None,
    ) -> RetrievalPlan:
        return RetrievalPlan(
            mode=mode,
            run_structured=run_structured,
            run_dense=hybrid,
            run_sparse=hybrid,
            run_hyde=(
                hybrid and self.settings.enable_hyde
                if run_hyde is None
                else run_hyde
            ),
            use_reranker=(
                hybrid and self.settings.enable_reranker
                if use_reranker is None
                else use_reranker
            ),
            reason=reason,
        )

    @staticmethod
    def _has_dental_signal(query: str) -> bool:
        normalized = normalize_vietnamese(query)
        return any(
            term in normalized
            for term in (
                "rang",
                "nha khoa",
                "nuou",
                "loi",
                "mieng",
                "implant",
                "chinh nha",
                "nho rang",
                "tay trang",
            )
        )
