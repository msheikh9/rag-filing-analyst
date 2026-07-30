import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, vector_size: int):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def upsert_points(self, points: list[qm.PointStruct]):
        """Insert/update vectors into Qdrant."""
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], limit: int):
        """Return the most similar chunks to the query vector."""
        logger.debug("Searching collection=%s limit=%d", self.collection, limit)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        logger.debug("Search returned %d results", len(results))
        return results


DENSE = "dense"
SPARSE = "sparse"


class HybridStore:
    """Qdrant collection holding a named dense vector and a named BM25 sparse vector.

    Sparse vectors cannot be added to a collection created with a single unnamed vector, so a
    hybrid collection has to be built fresh rather than migrated. Both legs are queried in one
    round trip and fused by Qdrant with Reciprocal Rank Fusion, which combines the two rankings
    by position instead of by score — dense cosine similarities and BM25 scores are on
    incomparable scales, so rank-based fusion avoids having to normalise between them.
    """

    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def recreate_collection(self, vector_size: int) -> None:
        """Drop and rebuild the collection with both vector fields."""
        if self.client.collection_exists(self.collection):
            logger.warning("Deleting existing collection %s", self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                # IDF is mandatory: fastembed ships term frequencies only, so without this
                # modifier Qdrant scores the sparse leg as a plain dot product.
                SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF),
            },
        )

    def upsert_points(self, points: list[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    @staticmethod
    def _stable_order(points: list) -> list:
        """Break score ties on chunk_id so equal-scoring results keep a fixed order.

        RRF produces exact ties routinely — a chunk ranked 1st by dense and 2nd by sparse fuses
        to the same score as one ranked 2nd then 1st — and Qdrant leaves the order of tied
        points unspecified, so it varies between identical requests. Left alone that makes the
        same question return differently ordered citations on a refresh, and makes evaluation
        results jitter (measured here: hit@1 varied by 0.04 across five identical runs).
        """
        return sorted(
            points,
            key=lambda p: (-p.score, str((p.payload or {}).get("chunk_id") or p.id)),
        )

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: qm.SparseVector,
        limit: int,
        prefetch_limit: int | None = None,
    ):
        """Fetch candidates from both legs and return the RRF-fused ranking."""
        prefetch_limit = prefetch_limit or limit
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=dense_vector, using=DENSE, limit=prefetch_limit),
                qm.Prefetch(query=sparse_vector, using=SPARSE, limit=prefetch_limit),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            # Over-fetch before the tie-break so that ties straddling the cut-off decide
            # membership deterministically too, not just ordering within the returned page.
            limit=limit * 2,
            with_payload=True,
        )
        return self._stable_order(response.points)[:limit]

    def dense_search(self, dense_vector: list[float], limit: int):
        """Dense-only query against the named vector — used for like-for-like comparison."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=dense_vector,
            using=DENSE,
            limit=limit * 2,
            with_payload=True,
        )
        return self._stable_order(response.points)[:limit]

    def sparse_search(self, sparse_vector: qm.SparseVector, limit: int):
        """Sparse-only query — used to check the BM25 leg carries signal on its own."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using=SPARSE,
            limit=limit * 2,
            with_payload=True,
        )
        return self._stable_order(response.points)[:limit]
