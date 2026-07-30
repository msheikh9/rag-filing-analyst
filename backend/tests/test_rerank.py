from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _reranker(scores):
    from src.rag_core.rerank import Reranker

    with patch("src.rag_core.rerank.CrossEncoder") as ce_cls:
        ce_cls.return_value = MagicMock()
        r = Reranker()
    r.model.predict.return_value = scores
    return r


class TestReranker:
    def test_sorts_by_descending_relevance(self):
        r = _reranker([-2.0, 6.0, 1.0])
        out = r.rerank("q", [{"text": "a"}, {"text": "b"}, {"text": "c"}])
        assert [c["text"] for c in out] == ["b", "c", "a"]

    def test_score_stays_within_zero_one(self):
        """A caller rendering score as a percentage must not see 605% for a logit of 6.05."""
        r = _reranker([6.05, -4.0])
        out = r.rerank("q", [{"text": "a"}, {"text": "b"}])
        assert all(0.0 <= c["score"] <= 1.0 for c in out)
        assert out[0]["score"] == pytest.approx(0.9976, abs=1e-4)

    def test_preserves_raw_logit_and_prior_score(self):
        r = _reranker([3.0])
        out = r.rerank("q", [{"text": "a", "score": 0.42}])
        assert out[0]["rerank_logit"] == 3.0
        assert out[0]["retrieval_score"] == 0.42

    def test_squashing_does_not_reorder(self):
        r = _reranker([0.5, 0.4, 10.0, -10.0])
        out = r.rerank("q", [{"text": f"t{i}"} for i in range(4)])
        assert [c["score"] for c in out] == sorted((c["score"] for c in out), reverse=True)

    def test_top_n_truncates(self):
        r = _reranker([1.0, 2.0, 3.0])
        out = r.rerank("q", [{"text": "a"}, {"text": "b"}, {"text": "c"}], top_n=2)
        assert len(out) == 2

    def test_empty_candidates_short_circuits(self):
        r = _reranker([])
        assert r.rerank("q", []) == []
        r.model.predict.assert_not_called()

    def test_does_not_mutate_input(self):
        r = _reranker([1.0])
        original = {"text": "a", "score": 0.5}
        r.rerank("q", [original])
        assert original == {"text": "a", "score": 0.5}
