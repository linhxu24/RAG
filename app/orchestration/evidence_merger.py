from decimal import Decimal
from typing import Any

from app.config import Settings
from app.orchestration.intent_registry import EntityScope, capability_for
from app.orchestration.schemas import BoundTaskPlan, EvidenceItem, EvidencePack

_TRUST_PRIORITY = {
    "authoritative": 0,
    "curated": 1,
    "retrieved": 2,
}


class EvidenceMerger:
    def __init__(self, settings: Settings):
        self.settings = settings

    def merge(
        self,
        *,
        query: str,
        plan: BoundTaskPlan,
        evidence: list[EvidenceItem],
    ) -> EvidencePack:
        task_priority = {task.task_id: task.priority for task in plan.tasks}
        ordered = sorted(
            evidence,
            key=lambda item: (
                task_priority.get(item.task_id, 999),
                _TRUST_PRIORITY.get(item.trust_level, 99),
                -item.score,
                item.source_type,
            ),
        )
        deduped = self._dedupe(ordered)
        limited = self._limit(deduped)
        return EvidencePack(
            query=query,
            tasks=plan.tasks,
            items=limited,
            conflicts=self._price_conflicts(limited),
            missing_info=self._missing_info(plan, limited),
        )

    def _limit(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        selected: list[EvidenceItem] = []
        source_counts: dict[str, int] = {}
        total_chars = 0
        for item in items:
            if len(selected) >= self.settings.max_evidence_items:
                break
            source_count = source_counts.get(item.source_type, 0)
            if (
                item.trust_level != "authoritative"
                and source_count >= self.settings.max_context_items_per_source
            ):
                continue
            if (
                selected
                and total_chars + len(item.text) > self.settings.max_context_chars
            ):
                continue
            selected.append(item)
            source_counts[item.source_type] = source_count + 1
            total_chars += len(item.text)
        return selected

    @staticmethod
    def _dedupe(items: list[EvidenceItem]) -> list[EvidenceItem]:
        seen_keys: set[str] = set()
        seen_text: set[str] = set()
        deduped: list[EvidenceItem] = []
        for item in items:
            canonical = item.canonical_key or f"{item.source_type}:{item.source_id}"
            key = f"{item.task_id}:{canonical}"
            normalized_text = f"{item.task_id}:{' '.join(item.text.lower().split())}"
            if key in seen_keys or normalized_text in seen_text:
                continue
            seen_keys.add(key)
            seen_text.add(normalized_text)
            deduped.append(item)
        return deduped

    @staticmethod
    def _missing_info(plan: BoundTaskPlan, items: list[EvidenceItem]) -> list[str]:
        missing = []
        for task in plan.tasks:
            task_items = [item for item in items if item.task_id == task.task_id]
            if task.clarification_required:
                missing.append(
                    task.clarification_question
                    or f"Bạn cần làm rõ yêu cầu cho: {task.effective_query}"
                )
                continue
            if task.resolved_ids and len(task_items) < len(task.resolved_ids):
                missing.append(
                    f"Chưa lấy đủ dữ liệu xác thực cho: {task.effective_query}"
                )
                continue
            capability = capability_for(task.intent)
            contract = capability.evidence_contract
            if capability.entity_scope == EntityScope.FILTER_ONLY:
                continue
            if capability.entity_scope == EntityScope.TWO_OR_MORE:
                satisfied = (
                    sum(
                        item.source_type in contract.allowed_source_types
                        for item in task_items
                    )
                    >= contract.minimum_items
                )
            elif contract.authoritative_required:
                satisfied = any(
                    item.source_type in contract.allowed_source_types
                    and item.trust_level == "authoritative"
                    for item in task_items
                )
            elif contract.allowed_source_types:
                satisfied = any(
                    item.source_type in contract.allowed_source_types
                    for item in task_items
                )
            else:
                satisfied = (
                    bool(task_items)
                    or not capability.allowed_tools
                )
            if not satisfied:
                missing.append(
                    f"Chưa có dữ liệu xác thực cho: {task.effective_query}"
                )
        return missing

    @staticmethod
    def _price_conflicts(items: list[EvidenceItem]) -> list[dict[str, Any]]:
        prices_by_name: dict[str, set[str]] = {}
        for item in items:
            name = str(item.raw_json.get("name") or item.raw_json.get("question") or "")
            price = item.raw_json.get("price")
            if not name or price is None:
                continue
            prices_by_name.setdefault(name, set()).add(_price_key(price))
        return [
            {"name": name, "prices": sorted(prices)}
            for name, prices in prices_by_name.items()
            if len(prices) > 1
        ]


def _price_key(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    return str(value)
