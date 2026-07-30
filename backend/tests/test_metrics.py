import pytest

from eval.metrics import (
    dcg_at_k,
    evaluate_run,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k({"a", "b"}, ["a", "b", "c"], 3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k({"a", "b"}, ["a", "x"], 2) == 0.5

    def test_cutoff_excludes_later_hits(self):
        assert recall_at_k({"a"}, ["x", "y", "a"], 2) == 0.0

    def test_empty_relevant_set_is_zero(self):
        assert recall_at_k(set(), ["a"], 5) == 0.0


class TestHitAtK:
    def test_any_hit_scores_one_regardless_of_relevant_count(self):
        # The distinction from recall: one of three relevant chunks still counts as a success.
        assert hit_at_k({"a", "b", "c"}, ["a"], 1) == 1.0
        assert recall_at_k({"a", "b", "c"}, ["a"], 1) == pytest.approx(1 / 3)

    def test_no_hit_scores_zero(self):
        assert hit_at_k({"a"}, ["x", "y"], 2) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "retrieved,expected",
        [
            (["a", "x", "y"], 1.0),
            (["x", "a", "y"], 0.5),
            (["x", "y", "a"], pytest.approx(1 / 3)),
        ],
    )
    def test_rank_positions(self, retrieved, expected):
        assert reciprocal_rank({"a"}, retrieved, 10) == expected

    def test_beyond_cutoff_is_zero(self):
        assert reciprocal_rank({"a"}, ["x"] * 10 + ["a"], 10) == 0.0

    def test_uses_first_relevant_only(self):
        assert reciprocal_rank({"a", "b"}, ["x", "b", "a"], 10) == 0.5


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k({"a", "b"}, ["a", "b", "c"], 10) == pytest.approx(1.0)

    def test_reversed_ranking_below_one(self):
        assert ndcg_at_k({"a", "b"}, ["x", "a", "b"], 10) < 1.0

    def test_ideal_accounts_for_relevant_count(self):
        # A single relevant chunk at rank 1 is a perfect ranking, so nDCG must be 1.0 even
        # though DCG is far below the two-relevant-chunk case.
        assert ndcg_at_k({"a"}, ["a", "b"], 10) == pytest.approx(1.0)

    def test_dcg_uses_log2_discount(self):
        assert dcg_at_k({"a"}, ["x", "a"], 10) == pytest.approx(1 / 1.5849625007, rel=1e-6)


class TestEvaluateRun:
    def test_missing_qid_scores_zero_rather_than_being_skipped(self):
        gold = {"q1": {"a"}, "q2": {"b"}}
        scores = evaluate_run({"q1": ["a"]}, gold, recall_ks=(1,))
        # q2 absent from the run must drag the average down, not be silently dropped.
        assert scores["recall@1"] == 0.5
        assert scores["mrr@10"] == 0.5

    def test_averages_over_queries(self):
        gold = {"q1": {"a"}, "q2": {"b"}}
        scores = evaluate_run({"q1": ["a"], "q2": ["x", "b"]}, gold, recall_ks=(1, 2))
        assert scores["recall@1"] == 0.5
        assert scores["recall@2"] == 1.0
        assert scores["mrr@10"] == pytest.approx(0.75)

    def test_empty_gold_returns_empty(self):
        assert evaluate_run({}, {}) == {}
