"""Compare saved eval runs: metric table, latency table, and per-query diagnostics.

Usage:
    python -m eval.compare
    python -m eval.compare --runs dense_legacy sparse hybrid --baseline dense_legacy
"""

import argparse
import json
from pathlib import Path

from eval.run_eval import RESULTS_DIR, load_gold

METRIC_ORDER = ("hit@1", "hit@3", "hit@5", "hit@10", "recall@10", "mrr@10", "ndcg@10")


def load_runs(names: list[str] | None) -> dict[str, dict]:
    paths = (
        [RESULTS_DIR / f"{n}.json" for n in names] if names else sorted(RESULTS_DIR.glob("*.json"))
    )
    runs: dict[str, dict] = {}
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing run file: {p}")
        runs[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def fmt_delta(value: float, base: float) -> str:
    d = value - base
    if abs(d) < 5e-5:
        return "   ="
    return f"{d:+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--baseline", default="dense_legacy")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    gold = {r["qid"]: set(r["relevant_chunk_ids"]) for r in load_gold()}
    gold_text = {r["qid"]: r["query"] for r in load_gold()}
    base = runs.get(args.baseline)

    name_w = max(len(n) for n in runs) + 2

    print("=" * (name_w + 9 * len(METRIC_ORDER)))
    print("METRICS")
    print("=" * (name_w + 9 * len(METRIC_ORDER)))
    header = "".join(f"{m:>10}" for m in METRIC_ORDER)
    print(f"{'run':<{name_w}}{header}")
    for name, data in runs.items():
        row = "".join(f"{data['metrics'].get(m, 0.0):>10.4f}" for m in METRIC_ORDER)
        print(f"{name:<{name_w}}{row}")

    if base:
        print(f"\ndelta vs {args.baseline}")
        for name, data in runs.items():
            if name == args.baseline:
                continue
            row = "".join(
                f"{fmt_delta(data['metrics'].get(m, 0.0), base['metrics'].get(m, 0.0)):>10}"
                for m in METRIC_ORDER
            )
            print(f"{name:<{name_w}}{row}")

    print("\n" + "=" * 60)
    print("LATENCY (ms, retrieval path only)")
    print("=" * 60)
    print(f"{'run':<{name_w}}{'p50':>10}{'p95':>10}{'rerank p50':>13}")
    for name, data in runs.items():
        total = data.get("retrieval_total_ms") or {}
        rr = (data.get("latency_ms") or {}).get("rerank", {})
        rr_s = f"{rr['p50_ms']:.1f}" if rr else "-"
        print(
            f"{name:<{name_w}}{total.get('p50_ms', 0):>10.1f}{total.get('p95_ms', 0):>10.1f}{rr_s:>13}"
        )

    print("\n" + "=" * 60)
    print("TOP-1 AGREEMENT BETWEEN RUNS (fraction of queries with same rank-1 chunk)")
    print("=" * 60)
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ra, rb = runs[a]["run"], runs[b]["run"]
            shared = set(ra) & set(rb)
            same = sum(1 for q in shared if ra[q] and rb[q] and ra[q][0] == rb[q][0])
            print(f"  {a:<26} vs {b:<26} {same}/{len(shared)}")

    print("\n" + "=" * 60)
    print("PER-QUERY: queries missed at rank 1")
    print("=" * 60)
    for name, data in runs.items():
        missed = [
            q for q, rel in gold.items() if not (data["run"].get(q) and data["run"][q][0] in rel)
        ]
        print(f"\n{name}  ({len(missed)} missed)")
        for q in sorted(missed):
            ranked = data["run"].get(q, [])
            rank = next((i for i, c in enumerate(ranked, 1) if c in gold[q]), None)
            print(f"  {q} rank={rank or '>20'}  {gold_text[q][:78]}")


if __name__ == "__main__":
    main()
