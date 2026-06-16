import time
import uuid
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import (
    EvaluationCaseResult,
    EvaluationRun,
    Product,
    RagTraceStep,
    Service,
)
from app.evaluation.diagnostics import build_diagnostics
from app.evaluation.eval_assets import evaluate_assets
from app.evaluation.eval_e2e import evaluate_e2e
from app.evaluation.eval_generation import evaluate_generation
from app.evaluation.eval_retrieval import (
    evaluate_reranker,
    evaluate_retrieval,
    retrieval_case_scores,
)
from app.evaluation.faithfulness import evaluate_answer
from app.evaluation.ground_truth import resolve_expected_ids
from app.evaluation.source_loader import (
    expected_asset_ids_for_sources,
    load_source_text,
)
from app.generation.schemas import ChatRequest
from app.services.chat import ChatService


def run_router_evaluation(
    session: Session,
    run: EvaluationRun,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.evaluation.eval_router import evaluate_router

    known_products = list(
        session.scalars(
            select(Product.name)
            .where(Product.status == "active")
            .order_by(Product.name)
        ).all()
    )
    known_services = list(
        session.scalars(
            select(Service.name)
            .where(Service.status == "active")
            .order_by(Service.name)
        ).all()
    )
    metrics = evaluate_router(
        cases,
        known_products=known_products,
        known_services=known_services,
    )
    predictions = metrics.pop("predictions", [])
    for case, prediction in zip(cases, predictions, strict=True):
        passed = prediction["expected"] == prediction["predicted"]
        session.add(
            EvaluationCaseResult(
                eval_run_id=run.eval_run_id,
                case_id=uuid.UUID(case["case_id"]),
                query=case["query"],
                expected_intent=prediction["expected"],
                actual_intent=prediction["predicted"],
                status="completed",
                passed=passed,
                scores={"intent_match": float(passed)},
                details={
                    "case_key": case.get("case_key"),
                    "confidence": prediction["confidence"],
                    "router_mode": "rule_baseline",
                },
            )
        )
    session.commit()
    return metrics


async def run_pipeline_evaluation(
    session: Session,
    settings: Settings,
    run: EvaluationRun,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    service = ChatService(settings)
    case_payloads: list[dict[str, Any]] = []
    retrieval_records: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    asset_records: list[dict[str, Any]] = []
    reranker_records: list[dict[str, Any]] = []
    trace_step_payloads: list[dict[str, Any]] = []

    for case in cases:
        resolution = resolve_expected_ids(session, case)
        started = time.perf_counter()
        result = EvaluationCaseResult(
            eval_run_id=run.eval_run_id,
            case_id=uuid.UUID(case["case_id"]),
            query=case["query"],
            expected_intent=case.get("expected_intent"),
            status="running",
            expected_ids=resolution.expected_ids,
        )
        session.add(result)
        session.commit()
        try:
            response = await service.chat(
                session,
                ChatRequest(
                    message=case["query"],
                    session_id=f"evaluation:{run.eval_run_id}:{case['case_id']}",
                ),
            )
            latency_ms = (time.perf_counter() - started) * 1000
            trace_id = uuid.UUID(response.trace_id)
            steps = session.scalars(
                select(RagTraceStep)
                .where(RagTraceStep.trace_id == trace_id)
                .order_by(RagTraceStep.created_at)
            ).all()
            step_payloads = [_step_payload(step) for step in steps]
            trace_step_payloads.extend(step_payloads)
            rankings = _rankings(step_payloads, response.model_dump(mode="json"))
            retrieved_ids = rankings["final"]
            source_text = load_source_text(session, retrieved_ids)
            response_payload = response.model_dump(mode="json")
            answer_scores, violations = evaluate_answer(
                answer_text=response.answer.text,
                answer_payload=response_payload,
                source_text=source_text,
                retrieved_ids=retrieved_ids,
                case=case,
            )
            retrieval_scores = retrieval_case_scores(
                retrieved_ids,
                resolution.expected_ids,
            )
            intent_match = (
                float(response.intent.value == case["expected_intent"])
                if case.get("expected_intent")
                else None
            )
            no_result = response.answer_type in {"fallback", "clarification"} or (
                "chưa có đủ thông tin" in response.answer.text.lower()
            )
            actual_retrieval_mode = _step_output(
                step_payloads,
                "retrieval_planning",
            ).get("mode")
            expected_retrieval_mode = case.get("metadata", {}).get(
                "expected_retrieval_mode"
            )
            retrieval_mode_match = (
                float(actual_retrieval_mode == expected_retrieval_mode)
                if expected_retrieval_mode
                else None
            )
            expected_no_result = bool(case.get("metadata", {}).get("expect_no_result"))
            no_result_match = float(no_result == expected_no_result) if expected_no_result else None
            if resolution.unresolved_keys:
                violations.append(
                    {
                        "type": "unresolved_ground_truth",
                        "values": resolution.unresolved_keys,
                    }
                )
            scores = {
                "intent_match": intent_match,
                **retrieval_scores,
                **answer_scores,
                "no_result_match": no_result_match,
                "retrieval_mode_match": retrieval_mode_match,
                "json_valid": 1.0,
                "schema_pass": 1.0,
            }
            if (
                response.intent.value in {"PRODUCT_LIST", "SERVICE_LIST"}
                and resolution.expected_ids
            ):
                retrieval_pass = set(resolution.expected_ids).issubset(retrieved_ids)
            else:
                retrieval_pass = (
                    retrieval_scores["hit_at_3"] == 1.0
                    if retrieval_scores["hit_at_3"] is not None
                    else None
                )
            scores["retrieval_pass"] = (
                float(retrieval_pass) if retrieval_pass is not None else None
            )
            required_scores = [
                scores.get("intent_match"),
                scores.get("answer_type_match"),
                scores.get("answer_correctness"),
                scores.get("faithfulness"),
                scores.get("safety_pass"),
                scores.get("no_result_match"),
                scores.get("retrieval_pass"),
                scores.get("retrieval_mode_match"),
            ]
            applicable = [value for value in required_scores if value is not None]
            passed = bool(applicable) and all(value >= 1.0 for value in applicable)
            if resolution.unresolved_keys:
                passed = False
            fallback = response.answer_type == "fallback" or any(
                step["step_name"] == "json_validation"
                and bool(step.get("output", {}).get("fallback"))
                for step in step_payloads
            )
            details = {
                "case_key": case.get("case_key"),
                "expected_answer_type": case.get("expected_answer_type"),
                "actual_answer_type": response.answer_type,
                "router_confidence": _step_output(
                    step_payloads,
                    "router_intent",
                ).get("confidence"),
                "router_needs_clarification": _step_output(
                    step_payloads,
                    "router_intent",
                ).get("needs_clarification"),
                "router_source": _step_output(
                    step_payloads,
                    "router_intent",
                ).get("source"),
                "expected_retrieval_mode": expected_retrieval_mode,
                "actual_retrieval_mode": actual_retrieval_mode,
                "expected_entities": case.get("expected_entities", []),
                "expected_source_keys": case.get("expected_source_keys", []),
                "unresolved_source_keys": resolution.unresolved_keys,
                "before_rerank_ids": rankings["before_rerank"],
                "after_rerank_ids": rankings["after_rerank"],
                "fallback": fallback,
                "no_result": no_result,
                "assets": response.answer.assets,
                "missing_assets": response.answer.missing_assets,
            }
            result.trace_id = trace_id
            result.actual_intent = response.intent.value
            result.status = "completed"
            result.passed = passed
            result.latency_ms = latency_ms
            result.retrieved_ids = retrieved_ids
            result.answer_text = response.answer.text
            result.scores = scores
            result.violations = violations
            result.details = details
            session.add(result)
            session.commit()

            case_payload = _result_payload(result)
            case_payloads.append(case_payload)
            retrieval_records.append(
                {
                    "expected_ids": resolution.expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "intent": response.intent.value,
                }
            )
            generation_records.append(scores)
            expected_asset_ids = (
                expected_asset_ids_for_sources(session, resolution.expected_ids)
                if case.get("metadata", {}).get("expect_assets")
                else [str(value) for value in case.get("expected_asset_ids", [])]
            )
            asset_records.append(
                {
                    "expected_asset_ids": expected_asset_ids,
                    "assets": response.answer.assets,
                }
            )
            reranker_records.append(
                {
                    "expected_ids": resolution.expected_ids,
                    "before_ids": rankings["before_rerank"],
                    "after_ids": rankings["after_rerank"],
                    "latency_ms": _step_latency(step_payloads, "reranker"),
                }
            )
        except Exception as exc:
            result.status = "failed"
            result.passed = False
            result.latency_ms = (time.perf_counter() - started) * 1000
            result.error_message = str(exc)
            result.violations = [{"type": "pipeline_error", "message": str(exc)}]
            session.add(result)
            session.commit()
            case_payloads.append(_result_payload(result))
            retrieval_records.append(
                {"expected_ids": resolution.expected_ids, "retrieved_ids": []}
            )
            generation_records.append({})
            asset_records.append({"expected_asset_ids": [], "assets": []})

    retrieval = evaluate_retrieval(retrieval_records)
    generation = evaluate_generation(generation_records)
    assets = evaluate_assets(asset_records)
    reranker = evaluate_reranker(reranker_records)
    router = _router_metrics(case_payloads)
    e2e = evaluate_e2e(case_payloads)
    per_intent = _per_intent(case_payloads)
    diagnostics = build_diagnostics(
        case_results=case_payloads,
        trace_steps=trace_step_payloads,
        retrieval_coverage=float(retrieval["ground_truth_coverage"]),
    )
    return {
        "router": router,
        "retrieval": retrieval,
        "reranker": reranker,
        "generation": generation,
        "assets": assets,
        "e2e": e2e,
        "coverage": _coverage(case_payloads),
        "per_intent": per_intent,
        "diagnostics": diagnostics,
    }


def _rankings(
    steps: list[dict[str, Any]],
    response: dict[str, Any],
) -> dict[str, list[str]]:
    by_name = {step["step_name"]: step for step in steps}

    def ids(step_name: str) -> list[str]:
        output = by_name.get(step_name, {}).get("output", {})
        return [
            str(item["id"])
            for item in output.get("results", [])
            if item.get("id")
        ]

    context_ids = [
        str(value)
        for value in by_name.get("context_builder", {})
        .get("output", {})
        .get("source_ids", [])
    ]
    response_ids = [
        str(item.get("source_id"))
        for item in response.get("answer", {}).get("sources", [])
        if item.get("source_id")
    ]
    structured = ids("structured_retrieval")
    rrf = ids("rrf_fusion")
    reranked = ids("reranker")
    combined = _unique(
        [
            *structured,
            *ids("dense_retrieval"),
            *ids("sparse_retrieval"),
        ]
    )
    before = rrf or combined
    after = reranked or before
    final = context_ids or response_ids or after or structured
    return {
        "final": _unique(final),
        "before_rerank": _unique(before),
        "after_rerank": _unique(after),
    }


def _router_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [item for item in results if item.get("expected_intent")]
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for item in applicable:
        confusion[item["expected_intent"]][item.get("actual_intent") or "ERROR"] += 1
    labels = sorted(
        {
            value
            for item in applicable
            for value in (
                item["expected_intent"],
                item.get("actual_intent") or "ERROR",
            )
        }
    )
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = sum(
            item["expected_intent"] == label and item.get("actual_intent") == label
            for item in applicable
        )
        false_positive = sum(
            item["expected_intent"] != label and item.get("actual_intent") == label
            for item in applicable
        )
        false_negative = sum(
            item["expected_intent"] == label and item.get("actual_intent") != label
            for item in applicable
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
    accuracy = (
        sum(item.get("actual_intent") == item["expected_intent"] for item in applicable)
        / len(applicable)
        if applicable
        else None
    )
    return {
        "case_count": len(applicable),
        "accuracy": accuracy,
        "per_label": per_label,
        "low_confidence_rate": (
            sum(
                float(item.get("details", {}).get("router_confidence") or 0) < 0.65
                for item in applicable
            )
            / len(applicable)
            if applicable
            else None
        ),
        "clarification_rate": (
            sum(
                bool(item.get("details", {}).get("router_needs_clarification"))
                for item in applicable
            )
            / len(applicable)
            if applicable
            else None
        ),
        "confusion_matrix": {
            expected: dict(predicted) for expected, predicted in confusion.items()
        },
    }


def _coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)

    def ratio(predicate) -> float:
        return sum(predicate(item) for item in results) / total if total else 0.0

    return {
        "case_count": total,
        "retrieval_ground_truth": ratio(lambda item: bool(item.get("expected_ids"))),
        "answer_ground_truth": ratio(
            lambda item: item.get("scores", {}).get("answer_correctness") is not None
        ),
        "faithfulness_applicable": ratio(
            lambda item: item.get("scores", {}).get("faithfulness") is not None
        ),
        "safety_applicable": ratio(
            lambda item: item.get("scores", {}).get("safety_pass") is not None
        ),
    }


def _per_intent(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[item.get("expected_intent") or "UNLABELED"].append(item)
    payload: dict[str, Any] = {}
    for intent, items in sorted(grouped.items()):
        retrieval_values = [
            item.get("scores", {}).get("recall_at_5")
            for item in items
            if item.get("scores", {}).get("recall_at_5") is not None
        ]
        faithfulness_values = [
            item.get("scores", {}).get("faithfulness")
            for item in items
            if item.get("scores", {}).get("faithfulness") is not None
        ]
        payload[intent] = {
            "case_count": len(items),
            "pass_rate": sum(item.get("passed") is True for item in items) / len(items),
            "router_accuracy": sum(
                item.get("actual_intent") == item.get("expected_intent") for item in items
            )
            / len(items),
            "retrieval_recall_at_5": (
                sum(retrieval_values) / len(retrieval_values)
                if retrieval_values
                else None
            ),
            "faithfulness_rate": (
                sum(faithfulness_values) / len(faithfulness_values)
                if faithfulness_values
                else None
            ),
        }
    return payload


def _step_payload(step: RagTraceStep) -> dict[str, Any]:
    return {
        "step_name": step.step_name,
        "status": step.status,
        "latency_ms": step.latency_ms,
        "input": step.input_json or {},
        "output": step.output_json or {},
        "error_message": step.error_message,
    }


def _step_latency(steps: list[dict[str, Any]], name: str) -> float:
    return float(
        next(
            (step.get("latency_ms", 0) for step in steps if step["step_name"] == name),
            0,
        )
    )


def _step_output(steps: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(
        (
            step.get("output", {})
            for step in steps
            if step["step_name"] == name
        ),
        {},
    )


def _result_payload(result: EvaluationCaseResult) -> dict[str, Any]:
    return {
        "result_id": str(result.result_id),
        "eval_run_id": str(result.eval_run_id),
        "case_id": str(result.case_id) if result.case_id else None,
        "trace_id": str(result.trace_id) if result.trace_id else None,
        "query": result.query,
        "expected_intent": result.expected_intent,
        "actual_intent": result.actual_intent,
        "status": result.status,
        "passed": result.passed,
        "latency_ms": result.latency_ms,
        "expected_ids": result.expected_ids or [],
        "retrieved_ids": result.retrieved_ids or [],
        "answer_text": result.answer_text,
        "scores": result.scores or {},
        "violations": result.violations or [],
        "details": result.details or {},
        "error_message": result.error_message,
    }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
