"""Index a JSONL of chunks into a Qdrant collection with named dense + BM25 sparse vectors.

Takes chunks from a file rather than re-deriving them from the dataset so that a retrieval
change and a chunking change are never measured at the same time: point this at the frozen
eval corpus to compare retrieval strategies, or at a re-chunked corpus to compare chunkings.

Usage:
    python -m scripts.index_hybrid --collection sec_filings_hybrid
    python -m scripts.index_hybrid --chunks eval/data/corpus_item.jsonl --collection sec_item
"""

import argparse
import uuid
from pathlib import Path

from qdrant_client.http import models as qm

from eval.corpus import CORPUS_PATH, load_corpus
from src.rag_core.config import settings
from src.rag_core.embeddings import Embedder
from src.rag_core.sparse import SparseEncoder
from src.rag_core.vectorstore import DENSE, SPARSE, HybridStore

PAYLOAD_FIELDS = (
    "doc_id",
    "chunk_id",
    "company",
    "year",
    "filingDate",
    "docID",
    "section",
    "text",
    "source",
    "sentence_ids",
    "n_sentences",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=CORPUS_PATH)
    ap.add_argument("--collection", default="sec_filings_hybrid")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    chunks = load_corpus(args.chunks)
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    embedder = Embedder(settings.embedding_model)
    sparse_encoder = SparseEncoder()
    store = HybridStore(url=settings.qdrant_url, collection=args.collection)

    vector_size = len(embedder.embed_query("test"))
    store.recreate_collection(vector_size=vector_size)
    print(f"Created '{args.collection}' with dense({vector_size}) + sparse(IDF)")

    total = 0
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        texts = [c["text"] for c in batch]

        dense_vecs = embedder.embed_texts(texts)
        sparse_vecs = sparse_encoder.embed_texts(texts)

        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector={DENSE: dense, SPARSE: sparse},
                payload={k: c[k] for k in PAYLOAD_FIELDS if k in c},
            )
            for c, dense, sparse in zip(batch, dense_vecs, sparse_vecs)
        ]
        store.upsert_points(points)
        total += len(points)
        print(f"  indexed {total}/{len(chunks)}")

    info = store.client.get_collection(args.collection)
    print(f"\nDone. '{args.collection}' holds {info.points_count} points.")


if __name__ == "__main__":
    main()
