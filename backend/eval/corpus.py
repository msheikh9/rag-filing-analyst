"""Corpus snapshot helpers.

The retrieval eval compares strategies (dense vs. hybrid vs. hybrid+rerank) over a
*frozen* set of chunks. Freezing the corpus to a JSONL file — rather than re-deriving it
from the Hugging Face dataset on every run — is what makes the before/after numbers
comparable: only the retrieval path changes between runs, never the documents.
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"


def dump_corpus(url: str, collection: str, out_path: Path = CORPUS_PATH) -> int:
    """Scroll every point out of a Qdrant collection into JSONL."""
    client = QdrantClient(url=url)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    offset = None
    with out_path.open("w", encoding="utf-8") as fh:
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                fh.write(json.dumps({"point_id": str(p.id), **payload}, ensure_ascii=False) + "\n")
                written += 1
            if offset is None:
                break

    return written


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    from src.rag_core.config import settings

    n = dump_corpus(settings.qdrant_url, settings.qdrant_collection)
    print(f"Dumped {n} chunks from '{settings.qdrant_collection}' to {CORPUS_PATH}")
