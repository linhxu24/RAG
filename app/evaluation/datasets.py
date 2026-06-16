import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.constants import Intent
from app.db.models import EvaluationCase, EvaluationDataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            cases.append(_normalize_case(payload, line_number))
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    keys = [case["case_key"] for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Evaluation dataset contains duplicate case_key values")
    return cases


def ensure_dataset(
    session: Session,
    *,
    path: Path,
    name: str = "dental_basic_eval",
    version: str = "2.0",
) -> tuple[EvaluationDataset, list[dict[str, Any]]]:
    cases = load_jsonl(path)
    content_hash = _content_hash(cases)
    dataset = session.scalar(
        select(EvaluationDataset).where(
            EvaluationDataset.name == name,
            EvaluationDataset.content_hash == content_hash,
        )
    )
    if dataset is not None:
        return dataset, _stored_cases(session, dataset.dataset_id)

    requested = session.scalar(
        select(EvaluationDataset).where(
            EvaluationDataset.name == name,
            EvaluationDataset.version == version,
        )
    )
    if requested is not None and requested.content_hash is None:
        dataset = requested
        dataset.content_hash = content_hash
        dataset.metadata_json = _dataset_metadata(cases, path)
        session.execute(
            delete(EvaluationCase).where(EvaluationCase.dataset_id == dataset.dataset_id)
        )
    else:
        effective_version = version if requested is None else f"{version}+{content_hash[:8]}"
        dataset = EvaluationDataset(
            name=name,
            version=effective_version,
            description=f"Imported from {path}",
            content_hash=content_hash,
            metadata_json=_dataset_metadata(cases, path),
        )
        session.add(dataset)
        session.flush()

    for case in cases:
        session.add(_case_model(dataset.dataset_id, case))
    session.commit()
    session.refresh(dataset)
    return dataset, _stored_cases(session, dataset.dataset_id)


def _normalize_case(case: dict[str, Any], line_number: int) -> dict[str, Any]:
    query = str(case.get("query", "")).strip()
    if not query:
        raise ValueError(f"Evaluation case at line {line_number} has no query")
    expected_intent = case.get("expected_intent")
    if expected_intent and expected_intent not in {item.value for item in Intent}:
        raise ValueError(
            f"Evaluation case at line {line_number} has invalid intent {expected_intent!r}"
        )
    expected_answer = case.get("expected_answer") or {}
    legacy_contains = expected_answer.get("contains") if isinstance(expected_answer, dict) else None
    expected_contains = case.get("expected_answer_contains", legacy_contains or [])
    if isinstance(expected_contains, str):
        expected_contains = [expected_contains]
    forbidden = case.get("forbidden_answer_contains", [])
    if isinstance(forbidden, str):
        forbidden = [forbidden]
    case_key = str(case.get("case_key") or _case_key(query, line_number))
    metadata = dict(case.get("metadata") or {})
    return {
        "case_key": case_key,
        "query": query,
        "expected_intent": expected_intent,
        "expected_answer_type": case.get("expected_answer_type"),
        "expected_doc_ids": case.get("expected_doc_ids") or [],
        "expected_chunk_ids": case.get("expected_chunk_ids") or [],
        "expected_row_ids": case.get("expected_row_ids") or [],
        "expected_asset_ids": case.get("expected_asset_ids") or [],
        "expected_entities": [str(item) for item in case.get("expected_entities", [])],
        "expected_source_keys": [str(item) for item in case.get("expected_source_keys", [])],
        "expected_answer_contains": [str(item) for item in expected_contains],
        "forbidden_answer_contains": [str(item) for item in forbidden],
        "expected_answer": expected_answer or None,
        "metadata": metadata,
    }


def _case_model(dataset_id: uuid.UUID, case: dict[str, Any]) -> EvaluationCase:
    return EvaluationCase(
        dataset_id=dataset_id,
        case_key=case["case_key"],
        query=case["query"],
        expected_intent=case.get("expected_intent"),
        expected_answer_type=case.get("expected_answer_type"),
        expected_doc_ids=_uuid_list(case.get("expected_doc_ids")),
        expected_chunk_ids=_uuid_list(case.get("expected_chunk_ids")),
        expected_row_ids=_uuid_list(case.get("expected_row_ids")),
        expected_asset_ids=_uuid_list(case.get("expected_asset_ids")),
        expected_entities=case.get("expected_entities", []),
        expected_source_keys=case.get("expected_source_keys", []),
        expected_answer_contains=case.get("expected_answer_contains", []),
        forbidden_answer_contains=case.get("forbidden_answer_contains", []),
        expected_answer=case.get("expected_answer"),
        metadata_json=case.get("metadata", {}),
    )


def _stored_cases(session: Session, dataset_id: uuid.UUID) -> list[dict[str, Any]]:
    records = session.scalars(
        select(EvaluationCase)
        .where(EvaluationCase.dataset_id == dataset_id)
        .order_by(EvaluationCase.case_key, EvaluationCase.case_id)
    ).all()
    return [
        {
            "case_id": str(item.case_id),
            "case_key": item.case_key,
            "query": item.query,
            "expected_intent": item.expected_intent,
            "expected_answer_type": item.expected_answer_type,
            "expected_doc_ids": [str(value) for value in item.expected_doc_ids or []],
            "expected_chunk_ids": [str(value) for value in item.expected_chunk_ids or []],
            "expected_row_ids": [str(value) for value in item.expected_row_ids or []],
            "expected_asset_ids": [str(value) for value in item.expected_asset_ids or []],
            "expected_entities": list(item.expected_entities or []),
            "expected_source_keys": list(item.expected_source_keys or []),
            "expected_answer_contains": list(item.expected_answer_contains or []),
            "forbidden_answer_contains": list(item.forbidden_answer_contains or []),
            "expected_answer": item.expected_answer,
            "metadata": item.metadata_json or {},
        }
        for item in records
    ]


def _dataset_metadata(cases: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    grounded = sum(
        bool(
            case["expected_source_keys"]
            or case["expected_doc_ids"]
            or case["expected_chunk_ids"]
            or case["expected_row_ids"]
        )
        for case in cases
    )
    answer_grounded = sum(bool(case["expected_answer_contains"]) for case in cases)
    return {
        "source_path": str(path),
        "case_count": len(cases),
        "retrieval_ground_truth_cases": grounded,
        "answer_ground_truth_cases": answer_grounded,
    }


def _content_hash(cases: list[dict[str, Any]]) -> str:
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _case_key(query: str, line_number: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:48]
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"{slug or f'case-{line_number}'}-{digest}"


def _uuid_list(values: list[str] | None) -> list[uuid.UUID] | None:
    if not values:
        return None
    return [uuid.UUID(value) for value in values]
