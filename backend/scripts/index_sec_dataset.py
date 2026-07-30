"""Index SEC 10-K filings into Qdrant with named dense + BM25 sparse vectors.

Chunking: sentences are packed to a token budget and never span a 10-K Item boundary. The
budget defaults to the embedding model's own maximum sequence length, because anything beyond
that is truncated before the dense vector is produced and would only ever be reachable through
the sparse leg.

Usage:
    python -m scripts.index_sec_dataset
    python -m scripts.index_sec_dataset --max-rows 20000 --max-tokens 256
"""

import argparse
import hashlib
import uuid

from datasets import load_dataset
from qdrant_client.http import models as qm

from src.rag_core.chunking import SentenceUnit, pack_units
from src.rag_core.config import settings
from src.rag_core.embeddings import Embedder
from src.rag_core.sparse import SparseEncoder
from src.rag_core.vectorstore import DENSE, SPARSE, HybridStore

DATASET = "khaihernlow/financial-reports-sec"
CONFIG = "small_lite"


def stable_id(*parts: str) -> str:
    raw = "||".join([p or "" for p in parts]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-rows", type=int, default=40000, help="sentences to read from the dataset"
    )
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = the embedder's max_seq_length")
    ap.add_argument("--overlap-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--collection", default=settings.qdrant_collection)
    args = ap.parse_args()

    ds = load_dataset(DATASET, CONFIG, split="train", trust_remote_code=True)

    # `section` is a ClassLabel, so the raw value is an integer index. Reading it directly and
    # falling back on falsiness maps section index 0 (Item 1, Business) onto "unknown" and
    # labels every other chunk with an index that looks like an Item number but is not one.
    int2str = ds.features["section"].int2str

    # Take a contiguous prefix rather than a shuffled sample: chunking needs consecutive
    # sentences from the same filing, and shuffling first would leave every chunk one sentence
    # long regardless of the token budget.
    ds = ds.select(range(min(args.max_rows, len(ds))))

    embedder = Embedder(settings.embedding_model)
    sparse_encoder = SparseEncoder(settings.sparse_model)
    store = HybridStore(url=settings.qdrant_url, collection=args.collection)

    max_tokens = args.max_tokens or embedder.model.max_seq_length
    tokenizer = embedder.model.tokenizer
    print(f"chunking to {max_tokens} tokens (overlap {args.overlap_tokens}), Item-aware")

    by_doc: dict[str, list[SentenceUnit]] = {}
    for row in ds:
        sentence = str(row.get("sentence") or "").strip()
        if not sentence:
            continue
        doc_id = str(row.get("docID") or "unknown")
        by_doc.setdefault(doc_id, []).append(
            SentenceUnit(
                sentence=sentence,
                n_tokens=len(tokenizer.tokenize(sentence)),
                section=int2str(row["section"]),
                sentence_id=str(row.get("sentenceID") or ""),
                meta={
                    "company": str(row.get("cik") or "unknown"),
                    "filingDate": str(row.get("filingDate") or "unknown"),
                },
            )
        )

    records: list[dict] = []
    for doc_id, units in by_doc.items():
        for idx, group in enumerate(
            pack_units(units, max_tokens, args.overlap_tokens, respect_sections=True)
        ):
            first = group[0]
            filing_date = first.meta["filingDate"]
            records.append(
                {
                    "doc_id": stable_id(doc_id),
                    "chunk_id": stable_id(doc_id, str(idx)),
                    "company": first.meta["company"],
                    "year": filing_date[:4] if len(filing_date) >= 4 else "unknown",
                    "filingDate": filing_date,
                    "docID": doc_id,
                    "section": first.section,
                    "text": " ".join(u.sentence for u in group),
                    "source": f"{DATASET}/{CONFIG}",
                    "sentence_ids": [u.sentence_id for u in group],
                    "n_sentences": len(group),
                }
            )

    print(f"{len(ds)} sentences from {len(by_doc)} filings -> {len(records)} chunks")

    store.recreate_collection(vector_size=len(embedder.embed_query("test")))

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        texts = [r["text"] for r in batch]
        dense_vecs = embedder.embed_texts(texts)
        sparse_vecs = sparse_encoder.embed_texts(texts)
        store.upsert_points(
            [
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={DENSE: dense, SPARSE: sparse},
                    payload=record,
                )
                for record, dense, sparse in zip(batch, dense_vecs, sparse_vecs)
            ]
        )
        print(f"  indexed {min(start + args.batch_size, len(records))}/{len(records)}")

    info = store.client.get_collection(args.collection)
    print(f"\nIndexed {info.points_count} chunks into '{args.collection}' (dense + sparse/IDF).")


if __name__ == "__main__":
    main()
