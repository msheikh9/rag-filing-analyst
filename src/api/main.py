from fastapi import FastAPI
from src.rag_core.config import settings
from src.rag_core.vectorstore import QdrantStore
from src.rag_core.embeddings import Embedder
from src.rag_core.llm import OllamaLLM
from .schemas import QueryRequest, QueryResponse, Citation

app = FastAPI(title="RAG Filing Analyst (Local)", version="0.1.0")

store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
embedder = Embedder(settings.embedding_model)
llm = OllamaLLM(settings.ollama_url, settings.ollama_model)


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    qvec = embedder.embed_query(req.query)
    results = store.search(query_vector=qvec, limit=settings.top_k)

    contexts = []
    citations: list[Citation] = []

    for r in results:
        p = r.payload or {}
        text = p.get("text", "")
        chunk_id = p.get("chunk_id")

        contexts.append(
            {
                "chunk_id": chunk_id,
                "text": text[:800],
            }
        )

        citations.append(
    Citation(
        score=float(r.score),
        chunk_id=chunk_id,
        company=p.get("company"),
        year=p.get("year"),
        filingDate=p.get("filingDate"),
        docID=p.get("docID"),
        section=p.get("section"),
        snippet=(text[:240] + "...") if text else None,
    )
)

    prompt = build_prompt(req.query, contexts)
    answer = llm.generate(prompt)

    return QueryResponse(answer=answer, citations=citations)