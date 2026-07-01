from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.config import Settings
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


@runtime_checkable
class ToolHandler(Protocol):
    def __call__(
        self,
        session: Session,
        task: BoundTask,
        tool_name: str,
    ) -> tuple[list[EvidenceItem], dict[str, Any] | None]:
        ...


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
        self._tool_registry: dict[str, ToolHandler] = {
            "product_tool": self._handle_structured_tool,
            "service_tool": self._handle_structured_tool,
            "clinic_info_tool": self._handle_structured_tool,
            "faq_tool": self._handle_faq_tool,
            "document_rag_tool": self._handle_document_rag,
        }

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
        handler = self._tool_registry.get(tool_name)
        if handler is None:
            raise ValueError(
                f"Tool '{tool_name}' is not registered in ToolExecutor. "
                f"Available tools: {sorted(self._tool_registry)}"
            )
        return handler(session, task, tool_name)

    def register_tool(
        self,
        tool_name: str,
        handler: ToolHandler,
        *,
        override: bool = False,
    ) -> None:
        """
        Register a runtime tool handler for extension/plugin tools.
        """
        if tool_name in self._tool_registry and not override:
            raise ValueError(
                f"Tool '{tool_name}' is already registered. "
                "Use override=True to replace."
            )
        self._tool_registry[tool_name] = handler

    def _handle_structured_tool(
        self,
        session: Session,
        task: BoundTask,
        tool_name: str,
    ) -> tuple[list[EvidenceItem], None]:
        return self._structured_evidence(session, task), None

    def _handle_faq_tool(
        self,
        session: Session,
        task: BoundTask,
        tool_name: str,
    ) -> tuple[list[EvidenceItem], None]:
        return self._faq_evidence(session, task), None

    def _handle_document_rag(
        self,
        session: Session,
        task: BoundTask,
        tool_name: str,
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        return self._document_rag_evidence(session, task)

    def _structured_evidence(
        self,
        session: Session,
        task: BoundTask,
    ) -> list[EvidenceItem]:
        results = self._structured_results(session, task)
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
    ) -> list[RetrievalResult]:
        resolved_ids = tuple(task.resolved_ids)
        capability = capability_for(task.intent)
        if resolved_ids and capability.entity_domain == "product":
            spec = ProductQuerySpec(
                product_ids=resolved_ids,
                limit=max(len(resolved_ids), 1),
            )
            return [
                self.structured._product_result(session, item, 1.0)
                for item in self.structured.list_products(session, spec)
            ]
        if resolved_ids and capability.entity_domain == "service":
            spec = ServiceQuerySpec(
                service_ids=resolved_ids,
                limit=max(len(resolved_ids), 1),
            )
            return [
                self.structured._service_result(session, item, 1.0)
                for item in self.structured.list_services(session, spec)
            ]
        if capability.entity_domain == "product":
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
        if capability.entity_domain == "service":
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
        if capability.primary_source_type == "clinic_info":
            return self.structured.clinic_info(session, self._search_query(task))
        return self.structured.retrieve(
            session,
            task.intent,
            self._search_query(task),
            list(task.entity_names),
        )

    def _faq_evidence(
        self,
        session: Session,
        task: BoundTask,
    ) -> list[EvidenceItem]:
        query = self._search_query(task)
        result_sets: dict[str, list[RetrievalResult]] = {
            "curated_faq": [
                self.structured._faq_result(faq, score)
                for faq, score in self.structured.search_faqs(
                    session,
                    query,
                    limit=self.settings.final_top_k,
                )
            ],
            "sparse_faq": self.sparse.retrieve_faqs(session, query),
            "dense_faq": self.dense.retrieve_faqs(session, query),
        }
        result_sets = {name: values for name, values in result_sets.items() if values}
        if not result_sets:
            return []
        fused = reciprocal_rank_fusion(
            result_sets,
            self.settings.rrf_k,
            {
                "curated_faq": self.settings.structured_rrf_weight,
                "sparse_faq": self.settings.sparse_rrf_weight,
                "dense_faq": self.settings.dense_rrf_weight,
            },
            max_per_source=self.settings.rrf_max_per_source,
        )[: self.settings.final_top_k]
        return [
            EvidenceItem.from_retrieval(
                task_id=task.task_id,
                result=item,
                trust_level=_trust_level(item),
            )
            for item in fused
        ]

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
        if capability_for(task.intent).primary_source_type == "faq":
            dense_sets.pop("dense_faq", None)
            sparse_sets.pop("sparse_faq", None)
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
