import pytest
from src.rag_core.chunking import pack_sentences


pytestmark = pytest.mark.unit


class TestPackSentences:
    def test_empty_list_returns_empty(self):
        assert pack_sentences([]) == []

    def test_single_short_sentence(self):
        result = pack_sentences(["Hello world."])
        assert result == ["Hello world."]

    def test_multiple_short_sentences_fit_in_one_chunk(self):
        sentences = ["First.", "Second.", "Third."]
        result = pack_sentences(sentences, max_chars=100)
        assert len(result) == 1
        assert result[0] == "First. Second. Third."

    def test_sentences_split_at_max_chars_boundary(self):
        sentences = ["A" * 50, "B" * 50, "C" * 50]
        result = pack_sentences(sentences, max_chars=110)
        assert len(result) == 2
        assert result[0] == ("A" * 50) + " " + ("B" * 50)
        assert result[1] == "C" * 50

    def test_single_sentence_exceeding_max_chars(self):
        long = "X" * 2000
        result = pack_sentences([long], max_chars=100)
        assert len(result) == 1
        assert result[0] == long

    def test_none_and_empty_strings_skipped(self):
        sentences = ["Valid.", None, "", "  ", "Also valid."]
        result = pack_sentences(sentences, max_chars=1000)
        assert result == ["Valid. Also valid."]

    def test_whitespace_stripped_from_sentences(self):
        result = pack_sentences(["  padded  ", "  text  "], max_chars=1000)
        assert result == ["padded text"]

    def test_default_max_chars_is_1800(self):
        sentences = ["word"] * 500
        result = pack_sentences(sentences)
        for chunk in result:
            assert len(chunk) <= 1800 or " " not in chunk

    def test_exact_boundary_fit(self):
        result = pack_sentences(["AAAA", "BBBB"], max_chars=10)
        assert result == ["AAAA BBBB"]

    def test_exact_boundary_overflow_by_one(self):
        result = pack_sentences(["AAAA", "BBBBB"], max_chars=9)
        assert result == ["AAAA", "BBBBB"]

    def test_many_tiny_sentences(self):
        sentences = ["w"] * 100
        result = pack_sentences(sentences, max_chars=20)
        for chunk in result:
            assert len(chunk) <= 20

    def test_all_empty_strings(self):
        assert pack_sentences(["", "", ""]) == []

    def test_all_none_values(self):
        assert pack_sentences([None, None]) == []

    def test_preserves_sentence_order(self):
        sentences = [f"S{i}" for i in range(10)]
        result = pack_sentences(sentences, max_chars=10)
        joined = " ".join(result)
        for i in range(10):
            assert f"S{i}" in joined
        positions = [joined.index(f"S{i}") for i in range(10)]
        assert positions == sorted(positions)
