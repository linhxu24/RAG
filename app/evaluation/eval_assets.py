from typing import Any


def evaluate_assets(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record.get("expected_asset_ids")]
    expected_total = resolved_total = missing_total = broken_total = wrong_total = 0
    for record in eligible:
        expected = {str(item) for item in record.get("expected_asset_ids", [])}
        assets = record.get("assets", [])
        resolved = {str(item.get("asset_id")) for item in assets}
        expected_total += len(expected)
        resolved_total += len(expected & resolved)
        missing_total += len(expected - resolved)
        wrong_total += len(resolved - expected)
        broken_total += sum(item.get("local_file_exists") is False for item in assets)
    return {
        "case_count": len(records),
        "eligible_case_count": len(eligible),
        "ground_truth_coverage": len(eligible) / len(records) if records else 0.0,
        "asset_resolve_success_rate": (
            resolved_total / expected_total if expected_total else None
        ),
        "missing_asset_rate": missing_total / expected_total if expected_total else None,
        "broken_local_file_rate": broken_total / expected_total if expected_total else None,
        "wrong_asset_rate": wrong_total / expected_total if expected_total else None,
    }
