"""Retrieval evaluation harness.

Runs one retrieval strategy over the gold set, reports Recall@k / MRR@10 / nDCG@10 and
per-stage latency percentiles, and writes the raw run to eval/results/ so any two strategies
can be compared after the fact.

Retrieval strategies:
    dense_legacy   single unnamed vector, client.search       (the pre-change baseline)
    dense          named dense vector only
    sparse         named BM25 sparse vector only
    hybrid         both legs, fused server-side with RRF

Cross-encoder reranking is an independent flag so it can be layered on any of them — which
leg benefits from reranking is an empirical question, not a property of hybrid search.

Usage:
    python -m eval.run_eval --strategy dense_legacy --collection sec_filings
    python -m eval.run_eval --strategy hybrid --collection sec_filings_hybrid --rerank --rerank-depth 30
"""

import argparse
import json
from pathlib import Path

from eval.metrics import evaluate_run
from eval.timing import StageTimer
from src.rag_core.config import settings
from src.rag_core.embeddings import Embedder
from src.rag_core.vectorstore import HybridStore, QdrantStore

EVAL_DIR = Path(__file__).resolve().parent
GOLD_PATH = EVAL_DIR / "gold" / "gold.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

STRATEGIES = ("dense_legacy", "dense", "sparse", "hybrid")


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dedupe(chunk_ids: list[str]) -> list[str]:
    """Collapse repeated chunk_ids, keeping first position.

    Identical boilerplate sentences share a chunk_id across points; leaving both in the ranking
    would push real results down a rank for no informational gain, and a user would see the
    same text twice.
    """
    seen: set[str] = set()
    out: list[str] = []
    for cid in chunk_ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--collection", default=settings.qdrant_collection)
    ap.add_argument(
        "--gold",
        type=Path,
        default=GOLD_PATH,
        help="gold file; use a chunking-projected gold when evaluating a re-chunked corpus",
    )
    ap.add_argument("--limit", type=int, default=50, help="candidates returned by retrieval")
    ap.add_argument("--prefetch-limit", type=int, default=50, help="candidates per hybrid leg")
    ap.add_argument("--rerank", action="store_true", help="rescore candidates with a cross-encoder")
    ap.add_argument("--rerank-depth", type=int, default=50, help="candidates fed to the reranker")
    ap.add_argument("--with-generation", action="store_true", help="also time Ollama generation")
    ap.add_argument("--gen-sample", type=int, default=10, help="queries to generate for")
    ap.add_argument("--gen-top-k", type=int, default=settings.top_k)
    ap.add_argument("--tag", default=None, help="results filename override")
    args = ap.parse_args()

    gold_rows = load_gold(args.gold)
    gold = {r["qid"]: set(r["relevant_chunk_ids"]) for r in gold_rows}
    timer = StageTimer()

    embedder = Embedder(settings.embedding_model)

    needs_sparse = args.strategy in ("sparse", "hybrid")
    sparse_encoder = None
    if needs_sparse:
        from src.rag_core.sparse import SparseEncoder

        sparse_encoder = SparseEncoder()

    reranker = None
    if args.rerank:
        from src.rag_core.rerank import Reranker

        reranker = Reranker()

    llm = None
    if args.with_generation:
        from src.rag_core.llm import OllamaLLM

        llm = OllamaLLM(settings.ollama_url, settings.ollama_model)

    if args.strategy == "dense_legacy":
        legacy = QdrantStore(settings.qdrant_url, args.collection)
        store = None
    else:
        store = HybridStore(settings.qdrant_url, args.collection)
        legacy = None

    # Warm every model before timing starts. The first forward pass of a torch model pays for
    # lazy kernel/graph setup and is several times slower than steady state; with only 50 queries
    # that single call would land squarely in the p95 and misreport the tail.
    warmup = "warmup query about consolidated revenue"
    embedder.embed_query(warmup)
    if sparse_encoder is not None:
        sparse_encoder.embed_query(warmup)
    if reranker is not None:
        reranker.rerank(warmup, [{"text": "warmup passage", "score": 0.0}])

    run: dict[str, list[str]] = {}
    print(f"Strategy={args.strategy} collection={args.collection} queries={len(gold_rows)}")

    for i, row in enumerate(gold_rows, start=1):
        qid, query = row["qid"], row["query"]

        dense_vec = None
        if args.strategy != "sparse":
            with timer.stage("embed_dense"):
                dense_vec = embedder.embed_query(query)

        sparse_vec = None
        if needs_sparse:
            with timer.stage("embed_sparse"):
                sparse_vec = sparse_encoder.embed_query(query)

        if args.strategy == "dense_legacy":
            with timer.stage("qdrant_dense"):
                points = legacy.search(query_vector=dense_vec, limit=args.limit)
        elif args.strategy == "dense":
            with timer.stage("qdrant_dense"):
                points = store.dense_search(dense_vec, limit=args.limit)
        elif args.strategy == "sparse":
            with timer.stage("qdrant_sparse"):
                points = store.sparse_search(sparse_vec, limit=args.limit)
        else:
            # Dense retrieval, sparse retrieval and RRF fusion all happen inside this one
            # request, so they cannot be timed apart from here; the dense and sparse strategies
            # give the per-leg cost.
            with timer.stage("qdrant_hybrid_fused"):
                points = store.hybrid_search(
                    dense_vec, sparse_vec, limit=args.limit, prefetch_limit=args.prefetch_limit
                )

        candidates = [
            {
                "chunk_id": (p.payload or {}).get("chunk_id"),
                "text": (p.payload or {}).get("text", ""),
                "score": float(p.score),
            }
            for p in points
        ]

        if reranker is not None:
            with timer.stage("rerank"):
                candidates = reranker.rerank(query, candidates[: args.rerank_depth])

        run[qid] = dedupe([c["chunk_id"] for c in candidates])

        if llm is not None and i <= args.gen_sample:
            context = "\n\n".join(c["text"] for c in candidates[: args.gen_top_k])
            prompt = (
                "Answer the question using ONLY the context below.\n\n"
                f"Question:\n{query}\n\nContext:\n{context}\n"
            )
            with timer.stage("generate"):
                llm.generate(prompt)

        if i % 10 == 0:
            print(f"  {i}/{len(gold_rows)} queries")

    scores = evaluate_run(run, gold)

    retrieval_stages = [
        "embed_dense",
        "embed_sparse",
        "qdrant_dense",
        "qdrant_sparse",
        "qdrant_hybrid_fused",
        "rerank",
    ]
    stage_stats = timer.percentiles()
    total = timer.total_per_query([s for s in retrieval_stages if s in stage_stats])

    print("\n--- metrics ---")
    for name, value in scores.items():
        print(f"  {name:<12} {value:.4f}")

    print("\n--- latency (ms) ---")
    for name, s in stage_stats.items():
        print(f"  {name:<22} p50 {s['p50_ms']:8.2f}   p95 {s['p95_ms']:8.2f}   n={s['n']}")
    if total:
        print(f"  {'RETRIEVAL TOTAL':<22} p50 {total['p50_ms']:8.2f}   p95 {total['p95_ms']:8.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (
        f"{args.strategy}_rerank{args.rerank_depth}" if args.rerank else args.strategy
    )
    out_path = RESULTS_DIR / f"{tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "strategy": args.strategy,
                "collection": args.collection,
                "config": {
                    "limit": args.limit,
                    "prefetch_limit": args.prefetch_limit,
                    "rerank_depth": args.rerank_depth if reranker else None,
                    "embedding_model": settings.embedding_model,
                },
                "metrics": scores,
                "latency_ms": stage_stats,
                "retrieval_total_ms": total,
                "run": {qid: ids[:20] for qid, ids in run.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
