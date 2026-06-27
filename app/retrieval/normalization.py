import re
import unicodedata

# Generic Vietnamese stopwords that provide no discriminative value for retrieval.
VIETNAMESE_SEARCH_STOPWORDS = {
    "bi",
    "cac",
    "can",
    "cho",
    "co",
    "cua",
    "da",
    "de",
    "do",
    "gi",
    "hay",
    "khong",
    "la",
    "lam",
    "ma",
    "mot",
    "nao",
    "nay",
    "nhu",
    "nhung",
    "o",
    "ra",
    "roi",
    "sao",
    "thi",
    "toi",
    "va",
    "ve",
    "voi",
}

# Domain terms that appear in almost every dental document.
# They hurt precision if kept as search tokens but must NOT be removed
# when the query would become empty after filtering.
DENTAL_DOMAIN_STOPWORDS = {
    "dich",
    "vu",
    "rang",
    "nha",
    "khoa",
    "nha khoa",
    "phong",
    "kham",
    "san",
    "pham",
}

SEARCH_STOP_PHRASES = (
    "cho toi",
    "do dau",
    "mat bao lau",
    "phai lam sao",
    "thong tin",
    "toi muon",
    "xin cho biet",
    "xu ly the nao",
)

SEARCH_SYNONYMS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("nhay", "cam"): ("e", "buot"),
    ("e", "buot"): ("nhay", "cam"),
    ("cao", "voi"): ("lay", "cao", "rang"),
    ("lay", "cao", "rang"): ("cao", "voi"),
    ("tay", "trang"): ("lam", "trang"),
    ("lam", "trang"): ("tay", "trang"),
    ("trong", "rang"): ("implant",),
    ("implant",): ("trong", "rang", "cay", "ghep"),
    ("nieng", "rang"): ("chinh", "nha"),
    ("chinh", "nha"): ("nieng", "rang"),
    ("nho", "rang"): ("nho",),
    ("sau", "rang"): ("lo", "rang"),
    ("lo", "rang"): ("sau", "rang"),
    ("chay", "mau"): ("chay", "mau", "chan", "rang"),
}


def normalize_vietnamese(value: str) -> str:
    """Remove diacritics, lowercase, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", value.lower())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", without_marks)).strip()


def normalize_for_match(text: str) -> str:
    """Shared normalizer for entity name matching across orchestration."""
    return normalize_vietnamese(text).lower().strip()


def query_tokens(value: str) -> list[str]:
    """Tokenize after normalization, keeping only tokens > 1 char."""
    return [
        token
        for token in normalize_vietnamese(value).split()
        if len(token) > 1
    ]


def search_query_tokens(value: str) -> list[str]:
    """Build search tokens: remove stopwords, expand synonyms.

    Synonym expansion runs against the full normalized string (before
    domain stopword removal) so that short tokens like ``e`` in
    ``ê buốt`` still trigger expansions.  Domain-specific stopwords are
    only removed when enough tokens remain afterwards.
    """
    normalized_full = normalize_vietnamese(value)

    search_text = normalized_full
    for phrase in SEARCH_STOP_PHRASES:
        search_text = re.sub(
            rf"\b{re.escape(phrase)}\b",
            " ",
            search_text,
        )

    base = [
        token
        for token in query_tokens(search_text)
        if token not in VIETNAMESE_SEARCH_STOPWORDS
    ]

    # Expand synonyms using the full normalized string (includes domain
    # words and single-char tokens that query_tokens would drop).
    expanded = list(base)
    for phrase, synonyms in SEARCH_SYNONYMS.items():
        if " ".join(phrase) in normalized_full:
            expanded.extend(token for token in synonyms if len(token) > 1)

    # Domain stopwords are only removed when enough tokens remain.
    filtered = [
        token for token in expanded if token not in DENTAL_DOMAIN_STOPWORDS
    ]
    if len(filtered) >= 2:
        expanded = filtered

    return list(dict.fromkeys(expanded))
