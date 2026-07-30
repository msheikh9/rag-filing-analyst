"""Retrieval orchestration: embed, search, optionally rerank.

Holds the pipeline in one place so the API layer does not branch on retrieval mode and so the
eval harness and the API exercise the same code path.
"""

import logging

from .config import Settings
from .embeddings import Embedder
from .vectorstore import HybridStore, QdrantStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mode = settings.retrieval_mode
        self.embedder = Embedder(settings.embedding_model)

        if self.mode == "hybrid":
            from .sparse import SparseEncoder

            self.store = HybridStore(settings.qdrant_url, settings.qdrant_collection)
            self.sparse_encoder = SparseEncoder(settings.sparse_model)
        elif self.mode == "dense":
            self.store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
            self.sparse_encoder = None
        else:
            raise ValueError(
                f"unknown RETRIEVAL_MODE: {self.mode!r} (expected 'hybrid' or 'dense')"
            )

        self.reranker = None
        if settings.enable_rerank:
            from .rerank import Reranker

            self.reranker = Reranker(settings.rerank_model)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Return up to `top_k` chunks as dicts carrying payload fields plus a score."""
        dense_vec = self.embedder.embed_query(query)

        # When reranking, pull a deeper candidate list than the caller asked for — the whole
        # point of the reranker is to promote chunks the first-stage ranking placed too low.
        depth = max(top_k, self.settings.rerank_depth) if self.reranker else top_k

        if self.mode == "hybrid":
            sparse_vec = self.sparse_encoder.embed_query(query)
            points = self.store.hybrid_search(
                dense_vec,
                sparse_vec,
                limit=depth,
                prefetch_limit=self.settings.prefetch_limit,
            )
        else:
            points = self.store.search(query_vector=dense_vec, limit=depth)

        candidates = []
        for p in points:
            payload = p.payload or {}
            candidates.append({**payload, "score": float(p.score)})

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)

        return candidates[:top_k]
