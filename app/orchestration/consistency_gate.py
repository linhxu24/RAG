from __future__ import annotations

from app.orchestration.intent_registry import EntityScope, capability_for
from app.orchestration.schemas import (
    BoundTask,
    BoundTaskPlan,
    ConsistencyReport,
    ConsistencyViolation,
    EvidenceItem,
    GateStatus,
)


class ConsistencyGate:
    """Mandatory firewall around tool execution and synthesis."""

    def check_bound_plan(self, plan: BoundTaskPlan) -> ConsistencyReport:
        violations: list[ConsistencyViolation] = []
        valid_task_ids: list[str] = []
        by_id = {task.task_id: task for task in plan.tasks}
        for task in plan.tasks:
            task_violations = self._task_violations(task, by_id)
            if task_violations:
                violations.extend(task_violations)
            elif not task.clarification_required:
                valid_task_ids.append(task.task_id)
        return ConsistencyReport(
            status=_status(violations),
            valid_task_ids=tuple(valid_task_ids),
            violations=tuple(violations),
        )

    def check_evidence(
        self,
        *,
        plan: BoundTaskPlan,
        evidence: list[EvidenceItem],
        valid_task_ids: tuple[str, ...] | None = None,
    ) -> ConsistencyReport:
        allowed_tasks = set(
            valid_task_ids
            if valid_task_ids is not None
            else (task.task_id for task in plan.tasks)
        )
        violations: list[ConsistencyViolation] = []
        passed: list[str] = []
        for task in plan.tasks:
            if task.task_id not in allowed_tasks:
                continue
            capability = capability_for(task.intent)
            contract = capability.evidence_contract
            task_items = [
                item
                for item in evidence
                if item.task_id == task.task_id
            ]
            if contract.minimum_items == 0 and not capability.allowed_tools:
                passed.append(task.task_id)
                continue
            if len(task_items) < contract.minimum_items:
                violations.append(
                    ConsistencyViolation(
                        task_id=task.task_id,
                        code="insufficient_evidence",
                        message=(
                            f"{task.intent.value} requires at least "
                            f"{contract.minimum_items} evidence item(s)"
                        ),
                        details={"actual": len(task_items)},
                    )
                )
                continue
            wrong_types = [
                item.source_type
                for item in task_items
                if contract.allowed_source_types
                and item.source_type not in contract.allowed_source_types
            ]
            if wrong_types:
                violations.append(
                    ConsistencyViolation(
                        task_id=task.task_id,
                        code="evidence_source_type_mismatch",
                        message="Evidence source type is not allowed for intent",
                        details={"source_types": wrong_types},
                    )
                )
                continue
            if contract.authoritative_required and task_items and not all(
                item.trust_level == "authoritative"
                for item in task_items
            ):
                violations.append(
                    ConsistencyViolation(
                        task_id=task.task_id,
                        code="authoritative_evidence_required",
                        message="Structured task received non-authoritative evidence",
                    )
                )
                continue
            if contract.match_resolved_ids:
                evidence_ids = {item.source_id for item in task_items}
                missing_ids = set(task.resolved_ids) - evidence_ids
                unexpected_ids = evidence_ids - set(task.resolved_ids)
                if missing_ids or unexpected_ids:
                    violations.append(
                        ConsistencyViolation(
                            task_id=task.task_id,
                            code="evidence_resolved_id_mismatch",
                            message="Evidence IDs do not match canonical resolved IDs",
                            details={
                                "resolved_ids": list(task.resolved_ids),
                                "evidence_ids": sorted(evidence_ids),
                                "missing_ids": sorted(missing_ids),
                                "unexpected_ids": sorted(unexpected_ids),
                            },
                        )
                    )
                    continue
            passed.append(task.task_id)
        return ConsistencyReport(
            status=_status(violations),
            valid_task_ids=tuple(passed),
            violations=tuple(violations),
        )

    @staticmethod
    def _task_violations(
        task: BoundTask,
        tasks_by_id: dict[str, BoundTask],
    ) -> list[ConsistencyViolation]:
        capability = capability_for(task.intent)
        values: list[ConsistencyViolation] = []
        entity_count = len(task.entity_names)
        id_count = len(task.resolved_ids)
        if capability.entity_scope == EntityScope.NONE and (
            entity_count or id_count
        ):
            values.append(
                _violation(
                    task,
                    "entity_blocked_for_intent",
                    "Intent must not carry product/service entity state",
                )
            )
        elif capability.entity_scope == EntityScope.FILTER_ONLY and (
            entity_count or id_count
        ):
            values.append(
                _violation(
                    task,
                    "filter_only_task_has_entity",
                    "List/filter intent must not carry resolved entity state",
                )
            )
        elif capability.entity_scope == EntityScope.EXACTLY_ONE and (
            entity_count != 1 or id_count != 1
        ):
            values.append(
                _violation(
                    task,
                    "entity_cardinality_mismatch",
                    "Detail task requires exactly one canonical entity and ID",
                    {
                        "entity_count": entity_count,
                        "resolved_id_count": id_count,
                    },
                )
            )
        elif capability.entity_scope == EntityScope.TWO_OR_MORE and (
            entity_count < 2 or id_count < 2 or entity_count != id_count
        ):
            values.append(
                _violation(
                    task,
                    "compare_cardinality_mismatch",
                    "Compare task requires two or more resolved entities",
                    {
                        "entity_count": entity_count,
                        "resolved_id_count": id_count,
                    },
                )
            )
        elif capability.entity_scope == EntityScope.OPTIONAL and (
            entity_count > 1 or id_count > 1 or entity_count != id_count
        ):
            values.append(
                _violation(
                    task,
                    "optional_entity_cardinality_mismatch",
                    "Optional entity scope allows zero or one resolved entity",
                    {
                        "entity_count": entity_count,
                        "resolved_id_count": id_count,
                    },
                )
            )
        if task.reference_mode not in capability.allowed_reference_modes:
            values.append(
                _violation(
                    task,
                    "reference_mode_not_allowed",
                    "Reference mode is not allowed by intent capability",
                    {"reference_mode": task.reference_mode.value},
                )
            )
        if task.entity_type and (
            task.entity_type not in capability.allowed_entity_types
        ):
            values.append(
                _violation(
                    task,
                    "entity_type_not_allowed",
                    "Entity type is not allowed by intent capability",
                    {"entity_type": task.entity_type},
                )
            )
        domain = capability.entity_domain
        filter_ids = (
            task.filters.product_ids
            if domain == "product"
            else task.filters.service_ids
            if domain == "service"
            else task.resolved_ids
        )
        if (
            capability.entity_scope != EntityScope.FILTER_ONLY
            and tuple(filter_ids) != tuple(task.resolved_ids)
        ):
            values.append(
                _violation(
                    task,
                    "filter_id_mismatch",
                    "Typed entity filters do not match resolved IDs",
                    {
                        "filter_ids": list(filter_ids),
                        "resolved_ids": list(task.resolved_ids),
                    },
                )
            )
        if task.inherited_from_task_id:
            source = tasks_by_id.get(task.inherited_from_task_id)
            if (
                source is None
                or source.resolution_status != "resolved"
                or source.clarification_required
                or not source.resolved_ids
            ):
                values.append(
                    _violation(
                        task,
                        "invalid_inheritance_source",
                        "Inherited task source is absent or not authoritatively resolved",
                        {
                            "source_task_id": task.inherited_from_task_id,
                        },
                    )
                )
        if task.clarification_required and not values:
            values.append(
                _violation(
                    task,
                    "task_requires_clarification",
                    task.clarification_question
                    or "Task requires clarification",
                )
            )
        return values


def _violation(
    task: BoundTask,
    code: str,
    message: str,
    details: dict | None = None,
) -> ConsistencyViolation:
    return ConsistencyViolation(
        task_id=task.task_id,
        code=code,
        message=message,
        details=details or {},
    )


def _status(
    violations: list[ConsistencyViolation],
) -> GateStatus:
    if not violations:
        return GateStatus.PASS
    if all(
        violation.code
        in {
            "task_requires_clarification",
            "entity_cardinality_mismatch",
            "compare_cardinality_mismatch",
            "optional_entity_cardinality_mismatch",
            "insufficient_evidence",
        }
        for violation in violations
    ):
        return GateStatus.CLARIFY
    return GateStatus.BLOCK
