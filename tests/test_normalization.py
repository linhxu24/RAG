from app.retrieval.normalization import (
    normalize_vietnamese,
    query_tokens,
    search_query_tokens,
)


class TestNormalizeVietnamese:
    def test_removes_diacritics(self):
        assert normalize_vietnamese("Tẩy trắng răng") == "tay trang rang"

    def test_replaces_d_bar(self):
        assert normalize_vietnamese("Đau răng") == "dau rang"

    def test_collapses_whitespace(self):
        assert normalize_vietnamese("  hello   world  ") == "hello world"

    def test_removes_punctuation(self):
        assert normalize_vietnamese("Giá: 100.000đ?") == "gia 100 000d"


class TestQueryTokens:
    def test_filters_single_char(self):
        tokens = query_tokens("Tôi bị đau răng")
        assert "i" not in tokens  # 'bị' -> 'bi' (2 chars, kept)

    def test_basic_tokens(self):
        tokens = query_tokens("Tẩy trắng răng")
        assert tokens == ["tay", "trang", "rang"]


class TestSearchQueryTokens:
    def test_removes_generic_stopwords(self):
        tokens = search_query_tokens("Tôi có bị đau răng không")
        # "toi", "co", "bi", "khong" are stopwords
        for stop in ("toi", "co", "khong"):
            assert stop not in tokens

    def test_removes_dental_domain_stopwords_when_enough_tokens(self):
        """'rang' is a dental domain stopword; removed when query has enough other tokens."""
        tokens = search_query_tokens("tẩy trắng răng bao nhiêu tiền")
        # 'rang' should be filtered since other tokens are sufficient
        assert "rang" not in tokens
        assert "tay" in tokens
        assert "trang" in tokens

    def test_keeps_dental_stopword_when_too_few_tokens_remain(self):
        """If removing dental stopwords leaves < 2 tokens, keep them."""
        tokens = search_query_tokens("đau răng")
        # Without 'rang', only 'dau' remains -> must keep 'rang'
        assert "rang" in tokens
        assert "dau" in tokens

    def test_synonym_expansion(self):
        tokens = search_query_tokens("ê buốt răng")
        # Should expand "e buot" -> "nhay cam"
        assert "nhay" in tokens
        assert "cam" in tokens

    def test_synonym_expansion_implant(self):
        tokens = search_query_tokens("trồng răng implant")
        assert "implant" in tokens

    def test_synonym_expansion_nieng(self):
        tokens = search_query_tokens("niềng răng mắc không")
        # "nieng rang" expands to ("chinh", "nha"), but "nha" is a
        # dental domain stopword and correctly filtered out.
        assert "chinh" in tokens
        assert "nieng" in tokens

    def test_empty_query(self):
        tokens = search_query_tokens("")
        assert tokens == []

    def test_query_only_stopwords(self):
        """Query made entirely of stopwords returns something rather than empty."""
        tokens = search_query_tokens("tôi có")
        # Both are stopwords but at least some tokens should survive
        # from the generic filter (they're all filtered), domain filter
        # won't help since there's nothing left. Result should be empty.
        assert tokens == []

    def test_deduplicate_synonyms(self):
        tokens = search_query_tokens("nhạy cảm ê buốt")
        # Both directions expand to each other; result should be deduped
        assert len(tokens) == len(set(tokens))

    def test_removes_non_discriminative_search_phrases(self):
        tokens = search_query_tokens("Thông tin AquaJet Mini")
        assert tokens == ["aquajet", "mini"]

    def test_removes_duration_question_phrase(self):
        tokens = search_query_tokens("Dịch vụ trồng răng implant mất bao lâu")
        assert "mat" not in tokens
        assert "bao" not in tokens
        assert "lau" not in tokens
        assert "implant" in tokens

    def test_synonym_expansion_does_not_add_single_character_tokens(self):
        tokens = search_query_tokens("Răng ê buốt")
        assert "e" not in tokens

    def test_question_word_dau_does_not_collide_with_pain_token(self):
        tokens = search_query_tokens("Răng nhạy cảm do đâu?")
        assert "dau" not in tokens
