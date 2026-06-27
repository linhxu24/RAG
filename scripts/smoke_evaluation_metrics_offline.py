"""Smoke-check offline evaluation metrics run without DB or LLM."""

from app.evaluation.eval_e2e import evaluate_e2e
from app.evaluation.eval_generation import evaluate_generation
from app.evaluation.eval_retrieval import evaluate_retrieval


def main() -> None:
    retrieval = evaluate_retrieval(
        [
            {
                "expected_ids": ["source-1"],
                "retrieved_ids": ["source-1", "source-2"],
            }
        ]
    )
    generation = evaluate_generation(
        [
            {
                "json_valid": 1.0,
                "schema_pass": 1.0,
                "faithfulness": 1.0,
                "answer_correctness": 1.0,
            }
        ]
    )
    e2e = evaluate_e2e(
        [
            {
                "status": "completed",
                "passed": True,
                "latency_ms": 12.0,
                "details": {},
            }
        ]
    )
    assert retrieval["hit_at_1"] == 1.0
    assert generation["schema_pass_rate"] == 1.0
    assert e2e["success_rate"] == 1.0
    print({"retrieval": retrieval, "generation": generation, "e2e": e2e})


if __name__ == "__main__":
    main()
