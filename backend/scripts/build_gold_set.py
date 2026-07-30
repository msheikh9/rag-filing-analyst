"""Stage 1 of gold-set construction: sample chunks and have Llama draft a query for each.

The drafts are NOT the gold set. Unedited synthetic queries reuse their source chunk's
vocabulary, which turns retrieval into string matching and inflates recall for every
strategy. Stage 2 is a manual rewrite pass over all 50 drafts (see eval/gold/gold.jsonl,
which keeps `draft_query` alongside the rewritten `query` so the edit is auditable).

Usage:
    python -m scripts.build_gold_set --n 50 --seed 17
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from eval.corpus import load_corpus
from src.rag_core.config import settings
from src.rag_core.llm import OllamaLLM

OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "data" / "gold_draft.jsonl"

# Chunks too short to contain an askable fact, or long enough to be a table dump, make bad
# gold anchors: the first are ambiguous, the second are answerable by many queries.
MIN_CHARS = 260
MAX_CHARS = 1400

# Sentences that appear verbatim across filings (signature blocks, exhibit indexes, generic
# accounting boilerplate) cannot anchor a *known-item* query — many chunks would answer it.
BOILERPLATE = re.compile(
    r"pursuant to the requirements of|incorporated (herein )?by reference|"
    r"see accompanying notes|table of contents|exhibit index|"
    r"signatures?$|/s/|no report is required|not applicable",
    re.IGNORECASE,
)

DRAFT_PROMPT = """You are helping build an evaluation set for a search engine over SEC 10-K filings.

Below is one passage from a filing. Write ONE question that this passage answers.

Rules:
- The question must be answerable from this passage alone.
- Ask about the substance: a figure, a policy, a risk, an event, an obligation.
- Write it the way an equity analyst would type it into a search box.
- One sentence. No preamble, no quotes, no explanation. Output only the question.

Passage:
{text}

Question:"""


def is_usable(row: dict) -> bool:
    text = (row.get("text") or "").strip()
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if BOILERPLATE.search(text):
        return False
    # Needs at least a couple of capitalised/numeric specifics to be worth asking about.
    return bool(re.search(r"\d", text))


def sample_chunks(corpus: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by section so the gold set is not 60% financial-statement footnotes."""
    rng = random.Random(seed)

    seen_ids: set[str] = set()
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in corpus:
        cid = row.get("chunk_id")
        if cid in seen_ids or not is_usable(row):
            continue
        seen_ids.add(cid)
        by_section[str(row.get("section"))].append(row)

    for rows in by_section.values():
        rng.shuffle(rows)

    # Round-robin across sections, largest pools last, until we have n.
    picked: list[dict] = []
    sections = sorted(by_section, key=lambda s: -len(by_section[s]))
    while len(picked) < n and any(by_section[s] for s in sections):
        for s in sections:
            if by_section[s] and len(picked) < n:
                picked.append(by_section[s].pop())

    rng.shuffle(picked)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    corpus = load_corpus()
    picked = sample_chunks(corpus, args.n, args.seed)
    print(f"Sampled {len(picked)} chunks from {len(corpus)} (seed={args.seed})")

    llm = OllamaLLM(settings.ollama_url, settings.ollama_model)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(picked, start=1):
            raw = llm.generate(DRAFT_PROMPT.format(text=row["text"]))
            query = raw.strip().strip('"').split("\n")[0].strip()
            record = {
                "qid": f"q{i:03d}",
                "draft_query": query,
                "relevant_chunk_ids": [row["chunk_id"]],
                "source_text": row["text"],
                "source_meta": {
                    "docID": row.get("docID"),
                    "company": row.get("company"),
                    "year": row.get("year"),
                    "section": row.get("section"),
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i:>2}/{len(picked)}] {query[:96]}")

    print(f"\nWrote drafts to {OUT_PATH}")
    print("Stage 2: rewrite every draft by hand into eval/gold/gold.jsonl before evaluating.")


if __name__ == "__main__":
    main()
