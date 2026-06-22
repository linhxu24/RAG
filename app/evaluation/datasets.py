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


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return load_json_dataset(path)
    return load_jsonl(path)


def load_json_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("Evaluation JSON must be an object")
    if "scenarios" in payload:
        return load_conversation_scenarios(path, payload=payload)
    if "groups" in payload:
        return load_semantic_groups(path, payload=payload)
    raise ValueError("Evaluation JSON must contain either groups or scenarios")


def load_semantic_groups(
    path: Path,
    *,
    payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if payload is None:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, list):
        raise ValueError("Semantic evaluation JSON must contain a groups array")
    cases: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            raise ValueError(f"Semantic group #{group_index} must be an object")
        group_key = str(group.get("case_group") or f"group_{group_index}")
        queries = group.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"Semantic group {group_key!r} has no queries")
        for query_index, query in enumerate(queries, start=1):
            case = {
                key: value
                for key, value in group.items()
                if key not in {"case_group", "queries", "expected_behavior"}
            }
            metadata = dict(case.get("metadata") or {})
            metadata["case_group"] = group_key
            if group.get("expected_behavior") is not None:
                metadata["expected_behavior"] = group["expected_behavior"]
            case["metadata"] = metadata
            case["case_key"] = f"{group_key}:{query_index:02d}"
            case["query"] = str(query)
            cases.append(_normalize_case(case, query_index))
    if not cases:
        raise ValueError("Semantic evaluation dataset is empty")
    return cases


def load_conversation_scenarios(
    path: Path,
    *,
    payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if payload is None:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list):
        raise ValueError("Conversation evaluation JSON must contain a scenarios array")
    cases: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise ValueError(f"Conversation scenario #{scenario_index} must be an object")
        scenario_key = str(scenario.get("scenario_key") or f"scenario_{scenario_index:02d}")
        turns = scenario.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Conversation scenario {scenario_key!r} has no turns")
        if not 4 <= len(turns) <= 6:
            raise ValueError(
                f"Conversation scenario {scenario_key!r} must contain 4-6 turns"
            )
        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"Conversation scenario {scenario_key!r} turn #{turn_index} "
                    "must be an object"
                )
            metadata = dict(scenario.get("metadata") or {})
            metadata.update(dict(turn.get("metadata") or {}))
            metadata["scenario_key"] = scenario_key
            metadata["scenario_title"] = scenario.get("title")
            metadata["turn_index"] = turn_index
            metadata["conversation_session_key"] = scenario_key
            case = {
                "case_key": f"{scenario_key}:turn_{turn_index:02d}",
                "query": turn.get("query"),
                "expected_intent": turn.get("expected_intent"),
                "expected_answer_type": turn.get("expected_answer_type"),
                "expected_entities": turn.get("expected_entities", []),
                "expected_source_keys": turn.get("expected_source_keys", []),
                "expected_answer_contains": turn.get("expected_answer_contains", []),
                "forbidden_answer_contains": turn.get("forbidden_answer_contains", []),
                "metadata": metadata,
            }
            cases.append(_normalize_case(case, turn_index))
    if not cases:
        raise ValueError("Conversation evaluation dataset is empty")
    return cases


def ensure_dataset(
    session: Session,
    *,
    path: Path,
    name: str = "dental_basic_eval",
    version: str = "2.0",
) -> tuple[EvaluationDataset, list[dict[str, Any]]]:
    cases = load_cases(path)
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
        "semantic_group_count": len(
            {
                case.get("metadata", {}).get("case_group")
                for case in cases
                if case.get("metadata", {}).get("case_group")
            }
        ),
        "conversation_scenario_count": len(
            {
                case.get("metadata", {}).get("scenario_key")
                for case in cases
                if case.get("metadata", {}).get("scenario_key")
            }
        ),
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
