"""Cross-encoder reranking of fused candidates.

The bi-encoder that produces the dense vectors never sees a query and a document together, so
it cannot model term interaction. A cross-encoder scores the pair jointly, which is far more
accurate and far more expensive — it is the dominant cost in the retrieval path, growing
linearly with the candidate count. Rerank depth is therefore the main quality/latency dial.
"""

import logging
import math

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, batch_size: int = 32):
        logger.info("Loading reranker: %s", model_name)
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        logger.info("Reranker loaded successfully.")

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        """Return `candidates` sorted by cross-encoder relevance, highest first.

        Each candidate must carry a "text" key. The raw logit is kept as "rerank_logit" and the
        pre-rerank score as "retrieval_score"; "score" is set to the squashed logit so that it
        stays a 0-1 relevance value whether or not reranking is enabled. ms-marco cross-encoders
        are trained with a logistic objective, so the sigmoid is the model's own calibrated
        probability rather than an arbitrary rescaling — and callers that render `score` as a
        percentage would otherwise show 605% for a logit of 6.05.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        ranked = []
        for c, s in zip(candidates, scores):
            item = dict(c)
            item.setdefault("retrieval_score", item.get("score"))
            logit = float(s)
            item["rerank_logit"] = logit
            # Sigmoid is monotonic, so squashing cannot reorder the results.
            item["score"] = 1.0 / (1.0 + math.exp(-logit))
            ranked.append(item)

        ranked.sort(key=lambda c: c["rerank_logit"], reverse=True)
        return ranked[:top_n] if top_n else ranked
