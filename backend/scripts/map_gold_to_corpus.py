"""Project the gold set onto a re-chunked corpus.

The gold set anchors each query to a source *sentence*. Chunk ids are a property of a chunking
scheme, so they cannot be compared across arms; sentence identity can. A chunk counts as
relevant when it contains the anchor sentence, which makes the same 50 queries scoreable
against any chunking without rewriting the gold set.

Usage:
    python -m scripts.map_gold_to_corpus --chunks eval/data/corpus_item_256_32.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from eval.corpus import load_corpus

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
DRAFT_PATH = EVAL_DIR / "data" / "gold_draft.jsonl"
GOLD_PATH = EVAL_DIR / "gold" / "gold.jsonl"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    corpus = load_corpus(args.chunks)
    with DRAFT_PATH.open(encoding="utf-8") as fh:
        drafts = {json.loads(line)["qid"]: json.loads(line) for line in fh if line.strip()}
    with GOLD_PATH.open(encoding="utf-8") as fh:
        gold = [json.loads(line) for line in fh if line.strip()]

    prepared = [(normalise(c["text"]), c["chunk_id"]) for c in corpus]

    out_path = args.out or (EVAL_DIR / "gold" / f"gold_{args.chunks.stem}.jsonl")
    unmatched: list[str] = []

    with out_path.open("w", encoding="utf-8") as fh:
        for row in gold:
            anchor = normalise(drafts[row["qid"]]["source_text"])
            hits = sorted(cid for text, cid in prepared if anchor and anchor in text)
            if not hits:
                unmatched.append(row["qid"])
            fh.write(
                json.dumps(
                    {**row, "relevant_chunk_ids": hits, "n_relevant": len(hits)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    matched = len(gold) - len(unmatched)
    print(f"{args.chunks.name}: matched {matched}/{len(gold)} anchors -> {out_path.name}")
    if unmatched:
        print(f"  unmatched (scored as unanswerable): {', '.join(unmatched)}")


if __name__ == "__main__":
    main()
