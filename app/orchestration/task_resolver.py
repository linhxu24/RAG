from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product, Service
from app.orchestration.intent_registry import detail_intent_for_entity_type
from app.orchestration.schemas import BindingDecision, PlannedTask, TaskResolution
from app.retrieval.entity_resolver import DatabaseEntityResolver, EntityCandidate


class TaskEntityResolver:
    """Resolve binding decisions to active authoritative database records."""

    def __init__(self, resolver: DatabaseEntityResolver):
        self.resolver = resolver

    def resolve(
        self,
        session: Session,
        *,
        task: PlannedTask,
        decision: BindingDecision,
    ) -> TaskResolution:
        if decision.entity_type not in {"product", "service"}:
            return TaskResolution(
                task_id=task.task_id,
                status="not_applicable",
                entity_type=decision.entity_type,
                entity_names=decision.entity_names,
            )

        verified = self._verify_inherited_ids(
            session,
            entity_type=decision.entity_type,
            ids=decision.inherited_resolved_ids,
        )
        verified_by_name = {candidate.name: candidate for candidate in verified}
        unresolved_names = [
            name
            for name in decision.entity_names
            if name not in verified_by_name
        ]
        selected = list(verified)
        candidates: list[EntityCandidate] = list(verified)
        ambiguous: list[EntityCandidate] = []
        statuses: list[str] = []

        detail_intent = detail_intent_for_entity_type(decision.entity_type)
        if detail_intent is None:
            return TaskResolution(
                task_id=task.task_id,
                status="not_applicable",
                entity_type=decision.entity_type,
                entity_names=decision.entity_names,
            )
        for name in unresolved_names:
            result = self.resolver.resolve(session, name, detail_intent)
            candidates.extend(result.candidates)
            ambiguous.extend(result.ambiguous_candidates)
            if result.status:
                statuses.append(result.status)
            selected.extend(result.selected)

        selected = _dedupe_candidates(selected)
        candidates = _dedupe_candidates(candidates)
        ambiguous = _dedupe_candidates(ambiguous)
        if ambiguous:
            status = "ambiguous"
        elif decision.clarification_required and not selected:
            status = "missing_context"
        elif selected and len(selected) == len(decision.entity_names):
            status = "resolved"
        elif selected:
            status = "partial"
        elif not decision.entity_names:
            status = "not_applicable"
        else:
            status = "not_found"
        return TaskResolution(
            task_id=task.task_id,
            status=status,
            entity_type=decision.entity_type,
            entity_names=tuple(candidate.name for candidate in selected),
            resolved_ids=tuple(candidate.entity_id for candidate in selected),
            candidates=tuple(
                candidate.as_dict() for candidate in candidates
            ),
            ambiguous_candidates=tuple(
                candidate.as_dict() for candidate in ambiguous
            ),
            source=(
                "conversation_state"
                if verified and not unresolved_names
                else "database"
            ),
        )

    @staticmethod
    def _verify_inherited_ids(
        session: Session,
        *,
        entity_type: str,
        ids: tuple[str, ...],
    ) -> list[EntityCandidate]:
        if not ids:
            return []
        model = Product if entity_type == "product" else Service
        id_column = (
            Product.product_id
            if entity_type == "product"
            else Service.service_id
        )
        rows = session.execute(
            select(id_column, model.name).where(
                model.status == "active",
                id_column.in_(ids),
            )
        ).all()
        by_id = {
            str(entity_id): str(name)
            for entity_id, name in rows
        }
        return [
            EntityCandidate(
                entity_type=entity_type,
                entity_id=entity_id,
                name=by_id[entity_id],
                score=1.0,
                match_type="verified_state_id",
            )
            for entity_id in ids
            if entity_id in by_id
        ]


def _dedupe_candidates(
    values: list[EntityCandidate],
) -> list[EntityCandidate]:
    deduped: dict[str, EntityCandidate] = {}
    for value in values:
        existing = deduped.get(value.entity_id)
        if existing is None or value.score > existing.score:
            deduped[value.entity_id] = value
    return list(deduped.values())
