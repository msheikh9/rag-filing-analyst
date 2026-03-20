from src.rag_core.config import settings
from src.rag_core.vectorstore import QdrantStore
from src.rag_core.embeddings import Embedder
from src.rag_core.llm import OllamaLLM


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        [f"[chunk_id={c['chunk_id']}] {c['text']}" for c in contexts]
    )
    return f"""You are a careful analyst.
Answer using ONLY the context below. If the context is insufficient, say you don't have enough information.

Question:
{question}

Context:
{context_block}

Return a concise answer and then a short list of cited chunk_id values you used.
"""


def main():
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    embedder = Embedder(settings.embedding_model)
    llm = OllamaLLM(settings.ollama_url, settings.ollama_model)

    question = "What kinds of risk factors are commonly mentioned in these filings?"
    qvec = embedder.embed_query(question)

    results = store.search(query_vector=qvec, limit=settings.top_k)

    contexts = []
    for r in results:
        p = r.payload or {}
        contexts.append(
            {
                "chunk_id": p.get("chunk_id"),
                "company": p.get("company"),
                "year": p.get("year"),
                "section": p.get("section"),
                "text": p.get("text", "")[:600],
                "score": float(r.score),
            }
        )

    prompt = build_prompt(question, contexts)
    answer = llm.generate(prompt)

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== CITATIONS (top retrieved chunks) ===\n")
    for c in contexts:
        print(
            f"- score={c['score']:.4f} chunk_id={c['chunk_id']} company={c['company']} year={c['year']} section={c['section']}"
        )


if __name__ == "__main__":
    main()