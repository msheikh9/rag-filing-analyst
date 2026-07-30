from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models as qm

from src.rag_core.config import Settings
from src.rag_core.vectorstore import DENSE, SPARSE, HybridStore

pytestmark = pytest.mark.unit


def _point(score, chunk_id, point_id="p"):
    return SimpleNamespace(score=score, id=point_id, payload={"chunk_id": chunk_id, "text": "t"})


class TestStableOrder:
    def test_sorts_by_descending_score(self):
        ordered = HybridStore._stable_order([_point(0.1, "a"), _point(0.9, "b")])
        assert [p.payload["chunk_id"] for p in ordered] == ["b", "a"]

    def test_breaks_ties_on_chunk_id(self):
        # RRF produces exact ties routinely; without a tiebreak the order is whatever Qdrant
        # happened to return, which varies between identical requests.
        forward = HybridStore._stable_order([_point(0.5, "zzz"), _point(0.5, "aaa")])
        reverse = HybridStore._stable_order([_point(0.5, "aaa"), _point(0.5, "zzz")])
        assert [p.payload["chunk_id"] for p in forward] == ["aaa", "zzz"]
        assert [p.payload["chunk_id"] for p in reverse] == ["aaa", "zzz"]

    def test_falls_back_to_point_id_when_chunk_id_missing(self):
        a = SimpleNamespace(score=0.5, id="b2", payload={})
        b = SimpleNamespace(score=0.5, id="a1", payload=None)
        assert [p.id for p in HybridStore._stable_order([a, b])] == ["a1", "b2"]


class TestHybridCollectionSchema:
    def test_declares_idf_modifier_on_sparse_field(self):
        """Without Modifier.IDF, Qdrant scores sparse vectors as a raw dot product."""
        store = HybridStore.__new__(HybridStore)
        store.client = MagicMock()
        store.collection = "c"
        store.client.collection_exists.return_value = False

        store.recreate_collection(vector_size=384)

        kwargs = store.client.create_collection.call_args.kwargs
        assert kwargs["vectors_config"][DENSE].size == 384
        assert kwargs["sparse_vectors_config"][SPARSE].modifier == qm.Modifier.IDF

    def test_drops_existing_collection_first(self):
        store = HybridStore.__new__(HybridStore)
        store.client = MagicMock()
        store.collection = "c"
        store.client.collection_exists.return_value = True

        store.recreate_collection(vector_size=384)
        store.client.delete_collection.assert_called_once_with("c")


class TestHybridSearch:
    def _store(self, points):
        store = HybridStore.__new__(HybridStore)
        store.client = MagicMock()
        store.collection = "c"
        store.client.query_points.return_value = SimpleNamespace(points=points)
        return store

    def test_queries_both_legs_and_fuses_with_rrf(self):
        store = self._store([_point(0.9, "a")])
        store.hybrid_search([0.1] * 384, qm.SparseVector(indices=[1], values=[1.0]), limit=10)

        kwargs = store.client.query_points.call_args.kwargs
        assert [p.using for p in kwargs["prefetch"]] == [DENSE, SPARSE]
        assert kwargs["query"].fusion == qm.Fusion.RRF

    def test_overfetches_then_truncates_to_limit(self):
        points = [_point(1.0 - i / 100, f"c{i:02d}") for i in range(20)]
        store = self._store(points)

        results = store.hybrid_search(
            [0.1] * 384, qm.SparseVector(indices=[1], values=[1.0]), limit=5
        )

        # Over-fetching is what makes tie-broken membership deterministic, not just ordering.
        assert store.client.query_points.call_args.kwargs["limit"] == 10
        assert len(results) == 5

    def test_prefetch_limit_defaults_to_limit(self):
        store = self._store([])
        store.hybrid_search([0.1] * 384, qm.SparseVector(indices=[1], values=[1.0]), limit=7)
        assert all(p.limit == 7 for p in store.client.query_points.call_args.kwargs["prefetch"])


class TestRetrieverConfig:
    def test_rejects_unknown_retrieval_mode(self):
        from src.rag_core.retriever import Retriever

        settings = Settings(retrieval_mode="magic")
        with patch("src.rag_core.retriever.Embedder"):
            with pytest.raises(ValueError, match="unknown RETRIEVAL_MODE"):
                Retriever(settings)

    def test_hybrid_mode_pulls_rerank_depth_when_reranking(self):
        from src.rag_core.retriever import Retriever

        settings = Settings(retrieval_mode="hybrid", enable_rerank=True, rerank_depth=30)
        with (
            patch("src.rag_core.retriever.Embedder"),
            patch("src.rag_core.sparse.SparseEncoder"),
            patch("src.rag_core.rerank.Reranker"),
            patch("src.rag_core.retriever.HybridStore") as store_cls,
        ):
            retriever = Retriever(settings)
            retriever.reranker.rerank.return_value = []
            store_cls.return_value.hybrid_search.return_value = []

            retriever.retrieve("q", top_k=5)

            # Reranking is pointless unless the candidate list is deeper than top_k.
            assert store_cls.return_value.hybrid_search.call_args.kwargs["limit"] == 30
