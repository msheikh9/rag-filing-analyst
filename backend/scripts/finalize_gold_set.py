"""Stage 2 of gold-set construction: merge the manual rewrites with the sampled anchors.

Also expands relevance beyond the single source chunk. The corpus contains sentences that
repeat verbatim across filings (2000 points share only 1993 distinct chunk_ids), and a query
answered by one copy is answered equally well by any other. Counting only the sampled copy
as relevant would penalise a retriever for surfacing an identical passage.

Usage:
    python -m scripts.finalize_gold_set
"""

import json
import re
from pathlib import Path

from eval.corpus import load_corpus

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
DRAFT_PATH = EVAL_DIR / "data" / "gold_draft.jsonl"
REWRITES_PATH = EVAL_DIR / "gold" / "rewrites.tsv"
GOLD_PATH = EVAL_DIR / "gold" / "gold.jsonl"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_rewrites(path: Path = REWRITES_PATH) -> dict[str, str]:
    rewrites: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            qid, _, query = line.partition("\t")
            rewrites[qid.strip()] = query.strip()
    return rewrites


def main() -> None:
    corpus = load_corpus()
    rewrites = load_rewrites()

    # Group chunk_ids by normalised text so duplicates can be pooled into the relevant set.
    by_text: dict[str, set[str]] = {}
    for row in corpus:
        by_text.setdefault(normalise(row["text"]), set()).add(row["chunk_id"])

    with DRAFT_PATH.open(encoding="utf-8") as fh:
        drafts = [json.loads(line) for line in fh if line.strip()]

    missing = [d["qid"] for d in drafts if d["qid"] not in rewrites]
    if missing:
        raise SystemExit(f"No rewrite supplied for: {', '.join(missing)}")

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    expanded = 0
    unchanged = 0

    with GOLD_PATH.open("w", encoding="utf-8") as fh:
        for d in drafts:
            qid = d["qid"]
            query = rewrites[qid]
            if normalise(query) == normalise(d["draft_query"]):
                unchanged += 1

            relevant = set(d["relevant_chunk_ids"])
            dupes = by_text.get(normalise(d["source_text"]), set())
            if len(dupes) > len(relevant):
                expanded += 1
            relevant |= dupes

            fh.write(
                json.dumps(
                    {
                        "qid": qid,
                        "query": query,
                        "relevant_chunk_ids": sorted(relevant),
                        "draft_query": d["draft_query"],
                        "source_meta": d["source_meta"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(drafts)} gold queries to {GOLD_PATH}")
    print(f"  queries whose relevant set grew via duplicate text: {expanded}")
    print(f"  queries left identical to the Llama draft: {unchanged}")


if __name__ == "__main__":
    main()
