from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


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
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )