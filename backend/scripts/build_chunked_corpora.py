"""Build the chunking A/B corpora from the Hugging Face dataset.

The dataset preserves 10-K Item boundaries as structured metadata, not as raw header text:
`section` is a ClassLabel over ['section_1', 'section_1A', ..., 'section_15'] and `sentenceID`
is '<docID>_section_<Item>_<index>'. So the Item-boundary arm splits on that label rather than
regex-matching "ITEM 1A. RISK FACTORS" out of the prose — same intent, exact boundaries.

Arms share an identical document set and token budget — only the boundary rule differs:
    fixed  pack sentences to the token budget across the whole filing, ignoring Items
    item   reset the buffer at every Item change, so no chunk spans two Items

`--sentence` instead emits one chunk per sentence, reproducing what the current production
indexer does. It exists so chunk size can be compared on the same documents: the shipped
sentence-level index was built from a different (smaller) document sample, so its metrics are
not comparable to the packed arms without rebuilding it here.

Usage:
    python -m scripts.build_chunked_corpora --max-tokens 256 --overlap-tokens 32
    python -m scripts.build_chunked_corpora --sentence
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from eval.corpus import DATA_DIR
from src.rag_core.chunking import SentenceUnit, pack_units
from src.rag_core.config import settings

DRAFT_PATH = Path(__file__).resolve().parent.parent / "eval" / "data" / "gold_draft.jsonl"
DATASET = "khaihernlow/financial-reports-sec"
CONFIG = "small_lite"


def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:24]


def gold_doc_ids() -> set[str]:
    with DRAFT_PATH.open(encoding="utf-8") as fh:
        return {json.loads(line)["source_meta"]["docID"] for line in fh if line.strip()}


def write_corpus(
    docs: dict[str, list[SentenceUnit]],
    out_path: Path,
    max_tokens: int,
    overlap_tokens: int,
    respect_sections: bool,
    source_tag: str,
) -> int:
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for doc_id, units in docs.items():
            for idx, group in enumerate(
                pack_units(units, max_tokens, overlap_tokens, respect_sections)
            ):
                text = " ".join(u.sentence for u in group)
                first = group[0]
                sections = sorted({u.section for u in group})
                record = {
                    "doc_id": stable_id(doc_id),
                    "chunk_id": stable_id(source_tag, doc_id, str(idx)),
                    "company": first.meta["cik"],
                    "year": first.meta["filingDate"][:4] or "unknown",
                    "filingDate": first.meta["filingDate"],
                    "docID": doc_id,
                    "section": sections[0] if len(sections) == 1 else "+".join(sections),
                    "text": text,
                    "source": f"{DATASET}/{CONFIG}",
                    "sentence_ids": [u.sentence_id for u in group],
                    "n_sentences": len(group),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--overlap-tokens", type=int, default=32)
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument(
        "--sentence",
        action="store_true",
        help="emit one chunk per sentence instead of the two packed arms",
    )
    args = ap.parse_args()

    ds = load_dataset(DATASET, CONFIG, split="train", trust_remote_code=True)
    int2str = ds.features["section"].int2str

    # Every filing that a gold query was drawn from must be present, or the query becomes
    # unanswerable and the A/B would measure corpus coverage instead of chunking.
    required = gold_doc_ids()
    all_docs = list(dict.fromkeys(ds["docID"]))
    fillers = [d for d in all_docs if d not in required]
    keep = set(required) | set(fillers[: max(0, args.n_docs - len(required))])
    print(f"gold docs={len(required)}  total docs kept={len(keep)} of {len(all_docs)}")

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

    docs: dict[str, list[SentenceUnit]] = {}
    for row in ds:
        doc_id = row["docID"]
        if doc_id not in keep:
            continue
        sentence = (row["sentence"] or "").strip()
        if not sentence:
            continue
        docs.setdefault(doc_id, []).append(
            SentenceUnit(
                sentence=sentence,
                n_tokens=len(tokenizer.tokenize(sentence)),
                section=int2str(row["section"]),
                sentence_id=row["sentenceID"],
                meta={"cik": row["cik"], "filingDate": str(row["filingDate"] or "unknown")},
            )
        )

    n_sentences = sum(len(v) for v in docs.values())
    print(f"kept {n_sentences} sentences across {len(docs)} filings")

    if args.sentence:
        # max_tokens=0 forces a flush after every sentence; overlap must be 0 or the previous
        # sentence would be re-seeded into the next chunk.
        out = DATA_DIR / "corpus_sentence.jsonl"
        n = write_corpus(docs, out, 0, 0, False, "sentence")
        print(f"  sentence -> {n:>6} chunks  {out.name}")
        return

    suffix = f"{args.max_tokens}_{args.overlap_tokens}"
    for tag, respect in (("fixed", False), ("item", True)):
        out = DATA_DIR / f"corpus_{tag}_{suffix}.jsonl"
        n = write_corpus(
            docs, out, args.max_tokens, args.overlap_tokens, respect, f"{tag}_{suffix}"
        )
        print(f"  {tag:<5} -> {n:>6} chunks  {out.name}")


if __name__ == "__main__":
    main()
