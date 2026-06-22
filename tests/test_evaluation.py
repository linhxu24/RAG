import json
from pathlib import Path

from app.evaluation.datasets import (
    load_cases,
    load_conversation_scenarios,
    load_jsonl,
    load_semantic_groups,
)
from app.evaluation.diagnostics import build_diagnostics
from app.evaluation.eval_assets import evaluate_assets
from app.evaluation.eval_conversation import (
    evaluate_conversation,
    evaluate_conversation_case,
)
from app.evaluation.eval_e2e import evaluate_e2e
from app.evaluation.eval_retrieval import evaluate_retrieval
from app.evaluation.faithfulness import evaluate_answer


def test_missing_retrieval_ground_truth_is_not_scored_as_perfect():
    metrics = evaluate_retrieval([{"retrieved_ids": [], "expected_ids": []}])
    assert metrics["ground_truth_coverage"] == 0.0
    assert metrics["recall_at_5"] is None
    assert metrics["ndcg_at_10"] is None


def test_missing_asset_ground_truth_is_not_scored():
    metrics = evaluate_assets([{"assets": [], "expected_asset_ids": []}])
    assert metrics["ground_truth_coverage"] == 0.0
    assert metrics["asset_resolve_success_rate"] is None


def test_dataset_loader_preserves_grounded_fields(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_key": "product-x",
                "query": "Sản phẩm X giá bao nhiêu?",
                "expected_intent": "PRODUCT_DETAIL",
                "expected_answer_type": "direct_data",
                "expected_entities": ["Sản phẩm X"],
                "expected_source_keys": ["product:Sản phẩm X"],
                "expected_answer_contains": ["100.000"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    case = load_jsonl(path)[0]
    assert case["expected_answer_type"] == "direct_data"
    assert case["expected_source_keys"] == ["product:Sản phẩm X"]
    assert case["expected_answer_contains"] == ["100.000"]


def test_semantic_group_loader_expands_queries(tmp_path):
    path = tmp_path / "semantic.json"
    path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "case_group": "product_price_filter",
                        "expected_intent": "PRODUCT_LIST",
                        "expected_behavior": {"all_items_price_lte": 2000000},
                        "queries": [
                            "Sản phẩm nào dưới 2 triệu?",
                            "Có món nào rẻ hơn 2.000.000 không?",
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_semantic_groups(path)

    assert [case["case_key"] for case in cases] == [
        "product_price_filter:01",
        "product_price_filter:02",
    ]
    assert all(case["expected_intent"] == "PRODUCT_LIST" for case in cases)
    assert cases[0]["metadata"]["case_group"] == "product_price_filter"
    assert cases[0]["metadata"]["expected_behavior"]["all_items_price_lte"] == 2000000


def test_conversation_scenario_loader_expands_turns_with_shared_session(tmp_path):
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_key": "whitening_followup",
                        "title": "Whitening follow-up",
                        "turns": [
                            {
                                "query": "Dịch vụ tẩy trắng răng giá bao nhiêu?",
                                "expected_intent": "SERVICE_DETAIL",
                            },
                            {
                                "query": "Mất bao lâu?",
                                "expected_intent": "SERVICE_DETAIL",
                            },
                            {
                                "query": "Có đau không?",
                                "expected_intent": "FAQ",
                            },
                            {
                                "query": "Cảm ơn",
                                "expected_intent": "CHITCHAT",
                            },
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_conversation_scenarios(path)

    assert [case["case_key"] for case in cases] == [
        "whitening_followup:turn_01",
        "whitening_followup:turn_02",
        "whitening_followup:turn_03",
        "whitening_followup:turn_04",
    ]
    assert {case["metadata"]["conversation_session_key"] for case in cases} == {
        "whitening_followup"
    }
    assert cases[1]["metadata"]["turn_index"] == 2


def test_committed_conversation_scenario_dataset_has_required_shape():
    cases = load_cases(Path("eval_datasets/dental_conversation_scenarios.json"))
    scenario_keys = {
        case["metadata"]["scenario_key"]
        for case in cases
    }
    turns_by_scenario = {
        key: [case for case in cases if case["metadata"]["scenario_key"] == key]
        for key in scenario_keys
    }

    assert len(scenario_keys) == 10
    assert all(4 <= len(turns) <= 6 for turns in turns_by_scenario.values())
    assert any(
        case["metadata"].get("requires_follow_up_memory")
        for case in cases
    )


def test_faithfulness_detects_unsupported_price():
    scores, violations = evaluate_answer(
        answer_text="Giá là 999.000đ",
        answer_payload={
            "answer_type": "direct_data",
            "answer": {"sources": [], "assets": [], "missing_assets": []},
            "safety": {},
        },
        source_text="Sản phẩm X. Giá: 100000",
        retrieved_ids=["product-1"],
        case={"expected_answer_type": "direct_data", "metadata": {}},
    )
    assert scores["faithfulness"] == 0.0
    assert any(item["type"] == "unsupported_price" for item in violations)


def test_diagnostics_flags_empty_retrieval_and_slow_generation():
    diagnostics = build_diagnostics(
        case_results=[
            {
                "expected_ids": ["expected"],
                "retrieved_ids": [],
                "passed": False,
                "details": {},
            }
        ],
        trace_steps=[
            {
                "step_name": "llm_generation",
                "latency_ms": 30_000,
                "status": "success",
            }
        ],
        retrieval_coverage=1.0,
    )
    codes = {item["code"] for item in diagnostics["alerts"]}
    assert "empty_retrieval" in codes
    assert "slow_llm_generation" in codes


def test_e2e_metrics_use_stored_case_results():
    metrics = evaluate_e2e(
        [
            {
                "status": "completed",
                "passed": True,
                "latency_ms": 10,
                "details": {
                    "fallback": False,
                    "no_result": False,
                    "actual_answer_type": "direct_data",
                },
            },
            {
                "status": "completed",
                "passed": True,
                "latency_ms": 30,
                "details": {
                    "fallback": False,
                    "no_result": True,
                    "actual_answer_type": "clarification",
                },
            },
        ]
    )

    assert metrics["success_rate"] == 1.0
    assert metrics["pass_rate"] == 1.0
    assert metrics["fallback_rate"] == 0.0
    assert metrics["no_result_rate"] == 0.5
    assert metrics["clarification_rate"] == 0.5


def test_conversation_case_scores_follow_up_binding_and_multi_task():
    scores, details, violations = evaluate_conversation_case(
        case={
            "expected_entities": ["Tẩy trắng răng"],
            "metadata": {
                "scenario_key": "whitening",
                "turn_index": 2,
                "requires_follow_up_memory": True,
                "expects_multi_task": True,
            },
        },
        trace_steps=[
            {
                "step_name": "memory_load",
                "output": {
                    "turn_count": 2,
                    "state": {"active_service_names": ["Tẩy trắng răng"]},
                },
            },
            {
                "step_name": "entity_span_extraction",
                "output": {
                    "provider": "gliner",
                    "degraded": False,
                    "spans": [],
                },
            },
            {
                "step_name": "context_binding",
                "output": {
                    "decisions": [
                        {
                            "binding_source": "conversation_state",
                            "entities_after": ["Tẩy trắng răng"],
                        }
                    ],
                    "plan": {
                        "tasks": [
                            {
                                "intent": "SERVICE_DETAIL",
                                "entities": ["Tẩy trắng răng"],
                                "selection": {
                                    "mentions": ["Tẩy trắng răng"],
                                    "resolution_status": "from_conversation_state",
                                },
                            },
                            {
                                "intent": "FAQ",
                                "entities": ["Tẩy trắng răng"],
                                "selection": {"mentions": ["Tẩy trắng răng"]},
                            },
                        ]
                    },
                },
            },
        ],
    )

    assert scores["entity_binding_match"] == 1.0
    assert scores["follow_up_memory"] == 1.0
    assert scores["multi_task_match"] == 1.0
    assert details["memory_context_used"] is True
    assert violations == []


def test_conversation_metrics_group_complete_scenarios():
    metrics = evaluate_conversation(
        [
            {
                "passed": True,
                "scores": {
                    "entity_binding_match": 1.0,
                    "follow_up_memory": None,
                    "multi_task_match": None,
                },
                "details": {
                    "conversation": {
                        "scenario_key": "whitening",
                        "scenario_title": "Tẩy trắng",
                        "turn_index": 1,
                        "entity_span_provider": "gliner",
                        "entity_span_degraded": False,
                    }
                },
            },
            {
                "passed": False,
                "scores": {
                    "entity_binding_match": 0.0,
                    "follow_up_memory": 0.0,
                    "multi_task_match": None,
                },
                "details": {
                    "conversation": {
                        "scenario_key": "whitening",
                        "scenario_title": "Tẩy trắng",
                        "turn_index": 2,
                        "entity_span_provider": "fallback",
                        "entity_span_degraded": True,
                    }
                },
            },
        ]
    )

    assert metrics["entity_binding_accuracy"] == 0.5
    assert metrics["follow_up_success_rate"] == 0.0
    assert metrics["scenario_pass_rate"] == 0.0
    assert metrics["entity_span_provider_counts"] == {"gliner": 1, "fallback": 1}
