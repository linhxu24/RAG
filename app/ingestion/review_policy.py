import re
from dataclasses import dataclass

_DYNAMIC_BUSINESS_REASON = re.compile(r"^table_\d+_row_\d+_")
_REVIEW_ONLY_EXACT = {
    "document_type_requires_review",
    "one_or_more_image_references_were_not_resolved",
    "review_required_by_upload_option",
}
_REVIEW_ONLY_SUFFIXES = (
    "_classification_low_confidence",
)


@dataclass(frozen=True)
class ReviewReasonSplit:
    review_only: list[str]
    integrity_blockers: list[str]


def split_review_reasons(
    reasons: list[str],
    *,
    ignore_dynamic_business_reasons: bool = False,
) -> ReviewReasonSplit:
    review_only: list[str] = []
    integrity: list[str] = []
    for reason in dict.fromkeys(reasons):
        if (
            ignore_dynamic_business_reasons
            and _DYNAMIC_BUSINESS_REASON.match(reason)
        ):
            # Business rows are validated again during approval.
            continue
        if reason in _REVIEW_ONLY_EXACT or reason.endswith(_REVIEW_ONLY_SUFFIXES):
            review_only.append(reason)
        else:
            integrity.append(reason)
    return ReviewReasonSplit(review_only=review_only, integrity_blockers=integrity)
