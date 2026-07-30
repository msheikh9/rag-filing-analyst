"""Paired significance testing between two saved runs.

With 50 queries a point-estimate gap of a few hundredths can be one or two queries changing
places. Both runs answer the *same* queries, so the comparison is paired: bootstrap over
queries and count how often the sign of the difference holds, rather than treating the two
averages as independent samples.

Usage:
    python -m eval.significance --baseline dense_legacy --candidate hybrid
    python -m eval.significance --baseline hybrid --candidate sparse --metric ndcg@10
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from eval.run_eval import RESULTS_DIR, load_gold

PER_QUERY = {
    "ndcg@10": lambda rel, ranked: ndcg_at_k(rel, ranked, 10),
    "mrr@10": lambda rel, ranked: reciprocal_rank(rel, ranked, 10),
    "hit@1": lambda rel, ranked: hit_at_k(rel, ranked, 1),
    "hit@3": lambda rel, ranked: hit_at_k(rel, ranked, 3),
}


def per_query_scores(
    run: dict[str, list[str]], gold: dict[str, set[str]], metric: str
) -> np.ndarray:
    fn = PER_QUERY[metric]
    return np.array([fn(rel, run.get(qid, [])) for qid, rel in sorted(gold.items())])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--metric", default="ndcg@10", choices=sorted(PER_QUERY))
    # Chunk ids are specific to a chunking scheme, so comparing two chunking arms means scoring
    # each run against its own projected gold. The comparison stays paired because both arms
    # answer the same 50 queries, keyed by qid.
    ap.add_argument("--gold", type=Path, default=None, help="gold used for both runs")
    ap.add_argument("--baseline-gold", type=Path, default=None)
    ap.add_argument("--candidate-gold", type=Path, default=None)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    base = json.loads((RESULTS_DIR / f"{args.baseline}.json").read_text())
    cand = json.loads((RESULTS_DIR / f"{args.candidate}.json").read_text())

    def _gold(path: Path | None) -> dict[str, set[str]]:
        rows = load_gold(path) if path else load_gold()
        return {r["qid"]: set(r["relevant_chunk_ids"]) for r in rows}

    base_gold = _gold(args.baseline_gold or args.gold)
    cand_gold = _gold(args.candidate_gold or args.gold)
    if set(base_gold) != set(cand_gold):
        raise SystemExit("the two gold files cover different qids; runs are not comparable")

    a = per_query_scores(base["run"], base_gold, args.metric)
    b = per_query_scores(cand["run"], cand_gold, args.metric)
    diff = b - a

    rng = np.random.default_rng(args.seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(args.iterations)])
    lo, hi = np.percentile(means, [2.5, 97.5])

    # Two-sided p-value by sign test over queries where the two runs actually differ.
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = n - wins - losses
    from math import comb

    m = wins + losses
    if m:
        k = min(wins, losses)
        p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2**m)
    else:
        p = 1.0

    print(f"metric   : {args.metric}")
    print(f"baseline : {args.baseline:<24} mean {a.mean():.4f}")
    print(f"candidate: {args.candidate:<24} mean {b.mean():.4f}")
    print(f"delta    : {diff.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  (paired bootstrap)")
    print(f"queries  : {wins} better, {losses} worse, {ties} unchanged   sign test p={p:.4f}")
    verdict = "significant at 0.05" if (lo > 0 or hi < 0) else "NOT significant at 0.05"
    print(f"verdict  : {verdict}")


if __name__ == "__main__":
    main()
