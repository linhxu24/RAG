"""Smoke-check trace summaries preserve fields consumed by evaluation."""

from app.observability.tracing import json_summary


def main() -> None:
    payload = {
        "source_ids": [f"id-{index}" for index in range(3)],
        "results": [
            {"id": f"id-{index}", "text": "x" * 200}
            for index in range(80)
        ],
        "extra": "y" * 10_000,
    }
    summary = json_summary(payload, max_chars=500)
    assert summary["truncated"] is True
    assert summary["source_ids"] == ["id-0", "id-1", "id-2"]
    assert len(summary["results"]) == 50
    assert summary["results_truncated_count"] == 30
    print(
        {
            "source_ids": summary["source_ids"],
            "results": len(summary["results"]),
            "truncated": summary["truncated"],
        }
    )


if __name__ == "__main__":
    main()
