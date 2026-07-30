"""BM25 sparse encoding via fastembed.

fastembed emits only the term-frequency half of BM25 — document values are TF components and
query values are all 1.0. The inverse-document-frequency factor is supplied by Qdrant when the
sparse vector field is declared with `Modifier.IDF`. Without that modifier Qdrant scores the
vectors as a raw dot product, which reduces the sparse leg to term counting with no rarity
weighting, so common words dominate and the BM25 leg quietly contributes almost nothing.
"""

import logging

from fastembed import SparseTextEmbedding
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)

DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class SparseEncoder:
    def __init__(self, model_name: str = DEFAULT_SPARSE_MODEL):
        logger.info("Loading sparse model: %s", model_name)
        self.model = SparseTextEmbedding(model_name)
        logger.info("Sparse model loaded successfully.")

    def embed_texts(self, texts: list[str]) -> list[qm.SparseVector]:
        return [
            qm.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self.model.embed(texts)
        ]

    def embed_query(self, text: str) -> qm.SparseVector:
        e = next(iter(self.model.query_embed(text)))
        return qm.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
