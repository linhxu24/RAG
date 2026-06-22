from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.constants import Intent
from app.orchestration.intent_registry import capability_for
from app.orchestration.schemas import (
    BoundTask,
    BoundTaskPlan,
    EvidenceItem,
    TrustLevel,
)
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.reranker import OptionalReranker
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.structured_query import ProductQuerySpec, ServiceQuerySpec
from app.retrieval.structured_retriever import StructuredRetriever
from app.retrieval.types import RetrievalResult


@dataclass
class ToolExecutionResult:
    evidence: list[EvidenceItem] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    reranker_runs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_count": len(self.evidence),
            "tool_counts": self.tool_counts,
            "errors": self.errors,
            "reranker_runs": self.reranker_runs,
            "evidence": [item.model_dump(mode="json") for item in self.evidence[:30]],
        }


class ToolExecutor:
    def __init__(
        self,
        *,
        structured: StructuredRetriever,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: OptionalReranker,
        settings: Settings,
    ) -> None:
        self.structured = structured
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.settings = settings

    def execute_many(
        self,
        session: Session,
        plan: BoundTaskPlan,
        *,
        valid_task_ids: tuple[str, ...] | None = None,
    ) -> ToolExecutionResult:
        result = ToolExecutionResult()
        allowed_task_ids = (
            set(valid_task_ids)
            if valid_task_ids is not None
            else {task.task_id for task in plan.tasks}
        )
        for task in sorted(plan.tasks, key=lambda item: item.priority):
            if task.task_id not in allowed_task_ids:
                continue
            if task.clarification_required:
                result.errors.append(
                    {
                        "task_id": task.task_id,
                        "tool": "all",
                        "error": task.clarification_question
                        or "task requires clarification",
                    }
                )
                continue
            for tool_name in capability_for(task.intent).allowed_tools:
                try:
                    evidence, rerank_meta = self._execute_tool(session, task, tool_name)
                except Exception as exc:
                    result.errors.append(
                        {
                            "task_id": task.task_id,
                            "tool": tool_name,
                            "error": str(exc),
                        }
                    )
                    continue
                result.tool_counts[tool_name] = result.tool_counts.get(tool_name, 0) + len(evidence)
                result.evidence.extend(evidence)
                if rerank_meta is not None:
                    result.reranker_runs.append(
                        {"task_id": task.task_id, **rerank_meta}
                    )
        return result

    def _execute_tool(
        self,
        session: Session,
        task: BoundTask,
        tool_name: str,
    ) -> tuple[list[EvidenceItem], dict[str, Any] | None]:
        if tool_name == "product_tool":
            intent = (
                task.intent
                if task.intent.name.startswith("PRODUCT_")
                else Intent.PRODUCT_DETAIL
            )
            return self._structured_evidence(session, task, intent), None
        if tool_name == "service_tool":
            intent = (
                task.intent
                if task.intent.name.startswith("SERVICE_")
                else Intent.SERVICE_DETAIL
            )
            return self._structured_evidence(session, task, intent), None
        if tool_name == "clinic_info_tool":
            return self._structured_evidence(session, task, Intent.CLINIC_INFO), None
        if tool_name == "faq_tool":
            return self._structured_evidence(session, task, Intent.FAQ), None
        if tool_name == "document_rag_tool":
            return self._document_rag_evidence(session, task)
        return [], None

    def _structured_evidence(
        self,
        session: Session,
        task: BoundTask,
        intent: Intent,
    ) -> list[EvidenceItem]:
        results = self._structured_results(session, task, intent)
        return [
            EvidenceItem.from_retrieval(
                task_id=task.task_id,
                result=item,
                trust_level=_trust_level(item),
            )
            for item in results
        ]

    def _structured_results(
        self,
        session: Session,
        task: BoundTask,
        intent: Intent,
    ) -> list[RetrievalResult]:
        resolved_ids = tuple(task.resolved_ids)
        if resolved_ids and intent in {
            Intent.PRODUCT_DETAIL,
            Intent.PRODUCT_COMPARE,
            Intent.PRODUCT_LIST,
        }:
            spec = ProductQuerySpec(
                product_ids=resolved_ids,
                limit=max(len(resolved_ids), 1),
            )
            return [
                self.structured._product_result(session, item, 1.0)
                for item in self.structured.list_products(session, spec)
            ]
        if resolved_ids and intent in {Intent.SERVICE_DETAIL, Intent.SERVICE_LIST}:
            spec = ServiceQuerySpec(
                service_ids=resolved_ids,
                limit=max(len(resolved_ids), 1),
            )
            return [
                self.structured._service_result(session, item, 1.0)
                for item in self.structured.list_services(session, spec)
            ]
        if intent == Intent.PRODUCT_LIST:
            from app.retrieval.structured_query import parse_product_query

            spec = parse_product_query(
                session,
                self._search_query(task),
                constraints=task.filters.as_constraints(),
                sort={
                    "field": task.filters.sort_field,
                    "direction": task.filters.sort_direction,
                },
                limit=task.filters.limit,
            )
            if spec.needs_clarification:
                return []
            return [
                self.structured._product_result(session, item, 1.0)
                for item in self.structured.list_products(session, spec)
            ]
        if intent == Intent.SERVICE_LIST:
            from app.retrieval.structured_query import parse_service_query

            spec = parse_service_query(
                session,
                self._search_query(task),
                constraints=task.filters.as_constraints(),
                sort={
                    "field": task.filters.sort_field,
                    "direction": task.filters.sort_direction,
                },
                limit=task.filters.limit,
            )
            if spec.needs_clarification:
                return []
            return [
                self.structured._service_result(session, item, 1.0)
                for item in self.structured.list_services(session, spec)
            ]
        return self.structured.retrieve(
            session,
            intent,
            self._search_query(task),
            list(task.entity_names),
        )

    def _document_rag_evidence(
        self,
        session: Session,
        task: BoundTask,
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        dense_sets = {
            f"dense_{name}": values
            for name, values in self.dense.retrieve_by_source(
                session,
                self._search_query(task),
                task.intent,
            ).items()
        }
        sparse_sets = {
            f"sparse_{name}": values
            for name, values in self.sparse.retrieve_by_source(
                session,
                self._search_query(task),
                task.intent,
            ).items()
        }
        result_sets = {**dense_sets, **sparse_sets}
        if not result_sets:
            return [], {"reranked": False, "reason": "no_results"}
        fused = reciprocal_rank_fusion(
            result_sets,
            self.settings.rrf_k,
            self._rrf_weights(result_sets),
            max_per_source=self.settings.rrf_max_per_source,
        )
        reranked, _, rerank_meta = self.reranker.rerank(
            self._search_query(task),
            fused,
        )
        return [
            EvidenceItem.from_retrieval(
                task_id=task.task_id,
                result=item,
                trust_level=_trust_level(item),
            )
            for item in reranked
        ], rerank_meta

    def _rrf_weights(
        self,
        result_sets: dict[str, list[RetrievalResult]],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name in result_sets:
            if name.startswith("sparse_"):
                weights[name] = self.settings.sparse_rrf_weight
            else:
                weights[name] = self.settings.dense_rrf_weight
        return weights

    @staticmethod
    def _search_query(task: BoundTask) -> str:
        return task.effective_query


def _trust_level(result: RetrievalResult) -> TrustLevel:
    if result.source_type in {"product", "service", "clinic_info"}:
        return "authoritative"
    if result.source_type == "faq":
        return "curated"
    return "retrieved"
