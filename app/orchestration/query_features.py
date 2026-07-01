from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from app.retrieval.normalization import normalize_vietnamese


@dataclass(frozen=True)
class QueryFeatures:
    is_short: bool
    is_social: bool
    is_schedule_query: bool
    is_availability: bool
    is_price: bool
    is_duration: bool
    asks_list: bool
    asks_compare: bool
    is_filter_refinement: bool
    is_question: bool
    has_entity_mention: bool
    has_context_reference: bool
    has_implicit_reference: bool

    _SOCIAL: ClassVar[frozenset[str]] = frozenset(
        {
            "cam on",
            "thank",
            "ok",
            "duoc roi",
            "hieu roi",
            "tam biet",
            "bye",
            "hen gap",
            "goodbye",
        }
    )
    _SCHEDULE: ClassVar[frozenset[str]] = frozenset(
        {
            "mo cua",
            "gio lam",
            "gio hoat dong",
            "thu may",
            "cuoi tuan",
            "chu nhat",
            "thu bay",
            "buoi sang",
            "buoi chieu",
            "lam viec",
            "hoat dong",
            "gio",
        }
    )
    _AVAILABILITY: ClassVar[frozenset[str]] = frozenset(
        {
            "con hang",
            "het hang",
            "con khong",
            "ton kho",
            "so luong",
            "het chua",
            "van con",
        }
    )
    _PRICE: ClassVar[frozenset[str]] = frozenset(
        {
            "gia",
            "bao nhieu",
            "chi phi",
            "phi",
            "tien",
            "gia ca",
            "gia ban",
            "gia tien",
        }
    )
    _DURATION: ClassVar[frozenset[str]] = frozenset(
        {
            "bao lau",
            "mat bao lau",
            "thoi gian",
            "lau khong",
            "mat may",
            "thoi luong",
        }
    )
    _QUESTION_WORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "co",
            "co the",
            "co duoc",
            "co phai",
            "sao",
            "nhu the nao",
            "tai sao",
            "khi nao",
            "duoc khong",
        }
    )
    _CONTEXT_REFERENCE: ClassVar[frozenset[str]] = frozenset(
        {
            "cai do",
            "san pham do",
            "dich vu do",
            "loai do",
            "cai nay",
            "loai nay",
            "no",
        }
    )
    _LIST: ClassVar[frozenset[str]] = frozenset(
        {
            "danh sach",
            "bang",
            "liet ke",
            "tat ca",
            "co nhung",
            "loai nao",
            "dich vu nao",
            "san pham nao",
        }
    )
    _COMPARE: ClassVar[frozenset[str]] = frozenset(
        {
            "so sanh",
            "khac nhau",
            "tot hon",
            "phu hop hon",
            " vs ",
        }
    )
    _FILTER_REFINEMENT: ClassVar[frozenset[str]] = frozenset(
        {
            "sap xep",
            "tang dan",
            "giam dan",
            "re nhat",
            "cao nhat",
            "trong so do",
            "con hang",
            "loc",
        }
    )

    @classmethod
    def extract(
        cls,
        query: str,
        entity_names_in_query: tuple[str, ...] = (),
    ) -> QueryFeatures:
        normalized = normalize_vietnamese(query).lower().strip()
        words = normalized.split()
        has_context_reference = _contains_any(normalized, cls._CONTEXT_REFERENCE)
        is_price = _contains_any(normalized, cls._PRICE)
        is_duration = _contains_any(normalized, cls._DURATION)
        is_availability = _contains_any(normalized, cls._AVAILABILITY)
        is_question = "?" in query or _contains_any(normalized, cls._QUESTION_WORDS)
        is_short = len(words) < 8
        return cls(
            is_short=is_short,
            is_social=_contains_any(normalized, cls._SOCIAL),
            is_schedule_query=_contains_any(normalized, cls._SCHEDULE),
            is_availability=is_availability,
            is_price=is_price,
            is_duration=is_duration,
            asks_list=_contains_any(normalized, cls._LIST),
            asks_compare=_contains_any(f" {normalized} ", cls._COMPARE),
            is_filter_refinement=(
                has_context_reference
                or _contains_any(normalized, cls._FILTER_REFINEMENT)
            ),
            is_question=is_question,
            has_entity_mention=bool(entity_names_in_query),
            has_context_reference=has_context_reference,
            has_implicit_reference=(
                has_context_reference
                or (
                    is_short
                    and (
                        is_question
                        or is_price
                        or is_duration
                        or is_availability
                    )
                )
            ),
        )


def _contains_any(normalized: str, keywords: frozenset[str]) -> bool:
    return any(keyword in normalized for keyword in keywords)
