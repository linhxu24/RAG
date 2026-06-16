import json

from app.evaluation.datasets import load_jsonl
from app.evaluation.diagnostics import build_diagnostics
from app.evaluation.eval_assets import evaluate_assets
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
