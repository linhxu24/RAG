from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.ner.entity_span_extractor import SpanExtractionResult
from app.orchestration.consistency_gate import ConsistencyGate
from app.orchestration.context_binder import ContextBinder
from app.orchestration.schemas import (
    BindingDecision,
    BoundTask,
    BoundTaskPlan,
    TaskPlan,
    TaskResolution,
)
from app.orchestration.task_canonicalizer import TaskCanonicalizer
from app.orchestration.task_resolver import TaskEntityResolver


@dataclass(frozen=True)
class BindingPipelineResult:
    bound_plan: BoundTaskPlan
    decisions: tuple[BindingDecision, ...]
    resolutions: tuple[TaskResolution, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": [
                decision.model_dump(mode="json")
                for decision in self.decisions
            ],
            "resolutions": [
                resolution.model_dump(mode="json")
                for resolution in self.resolutions
            ],
            "bound_plan": self.bound_plan.as_dict(),
        }


class TaskBindingPipeline:
    def __init__(
        self,
        *,
        binder: ContextBinder,
        resolver: TaskEntityResolver,
        canonicalizer: TaskCanonicalizer,
        consistency_gate: ConsistencyGate,
    ):
        self.binder = binder
        self.resolver = resolver
        self.canonicalizer = canonicalizer
        self.consistency_gate = consistency_gate

    def run(
        self,
        session: Session,
        *,
        plan: TaskPlan,
        original_query: str,
        history: dict[str, Any],
        span_result: SpanExtractionResult,
    ) -> BindingPipelineResult:
        decisions: list[BindingDecision] = []
        resolutions: list[TaskResolution] = []
        bound_tasks: list[BoundTask] = []
        inheritable_tasks: list[BoundTask] = []
        for task in sorted(plan.tasks, key=lambda item: item.priority):
            decision = self.binder.bind_task(
                task=task,
                original_query=original_query,
                history=history,
                span_result=span_result,
                prior_bound_tasks=tuple(inheritable_tasks),
            )
            resolution = self.resolver.resolve(
                session,
                task=task,
                decision=decision,
            )
            bound_task = self.canonicalizer.canonicalize(
                task=task,
                decision=decision,
                resolution=resolution,
                history=history,
            )
            decisions.append(decision)
            resolutions.append(resolution)
            bound_tasks.append(bound_task)
            source_report = self.consistency_gate.check_bound_plan(
                BoundTaskPlan(tasks=(bound_task,))
            )
            if source_report.passed:
                inheritable_tasks.append(bound_task)
        clarification = next(
            (
                task.clarification_question
                for task in bound_tasks
                if task.clarification_required
                and task.clarification_question
            ),
            plan.clarification_question,
        )
        return BindingPipelineResult(
            bound_plan=BoundTaskPlan(
                tasks=tuple(bound_tasks),
                clarification_question=clarification,
                metadata={
                    "planner_source": plan.source,
                    "planner_global_entities": list(
                        plan.planner_global_entities
                    ),
                },
            ),
            decisions=tuple(decisions),
            resolutions=tuple(resolutions),
        )
