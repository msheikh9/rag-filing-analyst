"""Retrieval metrics, computed from their definitions.

Relevance is binary: a chunk either answers the query or it does not. Gold queries may have
several relevant chunks, so Recall is over the full relevant set and nDCG's ideal ranking
places every relevant chunk first.
"""

import math
from collections.abc import Sequence


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """|relevant ∩ retrieved@k| / |relevant|"""
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def hit_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """1.0 if any relevant chunk appears in the top k, else 0.0.

    Recall is the wrong lens when a query has several acceptable chunks for reasons that have
    nothing to do with retrieval quality — chunk overlap duplicates an answer across
    neighbouring chunks, so Recall@1 is capped at 1/|relevant| and a perfect ranking still
    scores below 1. Hit rate asks the question the user actually cares about: did the answer
    show up at all.
    """
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def reciprocal_rank(relevant: set[str], retrieved: Sequence[str], k: int = 10) -> float:
    """1 / rank of the first relevant hit, or 0 if none appears in the top k."""
    for i, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevant: set[str], retrieved: Sequence[str], k: int = 10) -> float:
    return sum(
        1.0 / math.log2(i + 1)
        for i, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )


def ndcg_at_k(relevant: set[str], retrieved: Sequence[str], k: int = 10) -> float:
    """DCG@k / IDCG@k, where the ideal ranking front-loads every relevant chunk."""
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(relevant, retrieved, k) / ideal


def evaluate_run(
    run: dict[str, Sequence[str]],
    gold: dict[str, set[str]],
    recall_ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """Average each metric over the queries in `gold`.

    `run` maps qid -> ranked chunk_ids. A qid absent from `run` scores zero rather than being
    skipped, so a strategy cannot improve its average by failing to answer.
    """
    if not gold:
        return {}

    scores: dict[str, float] = {}
    for k in recall_ks:
        scores[f"recall@{k}"] = sum(
            recall_at_k(rel, run.get(qid, []), k) for qid, rel in gold.items()
        ) / len(gold)
    for k in recall_ks:
        scores[f"hit@{k}"] = sum(
            hit_at_k(rel, run.get(qid, []), k) for qid, rel in gold.items()
        ) / len(gold)

    scores["mrr@10"] = sum(
        reciprocal_rank(rel, run.get(qid, []), 10) for qid, rel in gold.items()
    ) / len(gold)
    scores["ndcg@10"] = sum(
        ndcg_at_k(rel, run.get(qid, []), 10) for qid, rel in gold.items()
    ) / len(gold)
    return scores
