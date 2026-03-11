import hashlib
import uuid
from datasets import load_dataset
from qdrant_client.http import models as qm

from src.rag_core.config import settings
from src.rag_core.vectorstore import QdrantStore
from src.rag_core.embeddings import Embedder
from src.rag_core.chunking import pack_sentences


def stable_id(*parts: str) -> str:
    raw = "||".join([p or "" for p in parts]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def main():
    # Load dataset from Hugging Face
    ds = load_dataset(
    "khaihernlow/financial-reports-sec",
    "small_lite",
    split="train",
    trust_remote_code=True,
)
    # Keep first run fast and cheap: index a small subset
    max_rows = 300
    ds = ds.select(range(min(max_rows, len(ds))))

    embedder = Embedder(settings.embedding_model)
    store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)

    # Get vector size and create collection if needed
    test_vec = embedder.embed_query("test")
    store.ensure_collection(vector_size=len(test_vec))

    points = []
    batch_texts = []
    batch_payloads = []

    for row in ds:
        company = str(row.get("cik") or row.get("company") or "unknown")
        year = str(row.get("year") or row.get("filing_year") or "unknown")
        section = str(row.get("section") or row.get("section_name") or "unknown")
        sentence = str(row.get("sentence") or row.get("text") or "").strip()
        if not sentence:
            continue

        # Pack sentence(s) into a chunk (simple first version)
        chunks = pack_sentences([sentence], max_chars=1800)

        for idx, chunk_text in enumerate(chunks):
            doc_id = stable_id(company, year, section)
            chunk_id = stable_id(doc_id, str(idx), chunk_text[:80])

            payload = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "company": company,
                "year": year,
                "section": section,
                "text": chunk_text,
                "source": "khaihernlow/financial-reports-sec",
            }

            batch_texts.append(chunk_text)
            batch_payloads.append(payload)

            if len(batch_texts) >= 64:
                vecs = embedder.embed_texts(batch_texts)
                for v, p in zip(vecs, batch_payloads):
                    points.append(qm.PointStruct(id=str(uuid.uuid4()), vector=v, payload=p))
                batch_texts, batch_payloads = [], []

    # Flush remaining
    if batch_texts:
        vecs = embedder.embed_texts(batch_texts)
        for v, p in zip(vecs, batch_payloads):
            points.append(qm.PointStruct(id=str(uuid.uuid4()), vector=v, payload=p))

    if points:
        store.upsert_points(points)

    print(f"Indexed {len(points)} chunks into Qdrant collection '{settings.qdrant_collection}'.")


if __name__ == "__main__":
    main()