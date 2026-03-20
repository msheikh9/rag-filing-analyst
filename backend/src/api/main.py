import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.rag_core.config import settings
from src.rag_core.vectorstore import QdrantStore
from src.rag_core.embeddings import Embedder
from src.rag_core.llm import OllamaLLM
from .schemas import QueryRequest, QueryResponse, Citation

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG dependencies...")
    app.state.store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
    app.state.embedder = Embedder(settings.embedding_model)
    app.state.llm = OllamaLLM(settings.ollama_url, settings.ollama_model)
    logger.info("RAG dependencies ready.")
    yield
    logger.info("Shutting down RAG Filing Analyst API.")


app = FastAPI(
    title="RAG Filing Analyst API",
    version="1.0.0",
    description="AI-powered SEC filing analysis with Retrieval-Augmented Generation",
    contact={
        "name": "RAG Filing Analyst",
        "url": "https://github.com/your-username/rag-filing-analyst",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        [f"[chunk_id={c['chunk_id']}] {c['text']}" for c in contexts]
    )
    return f"""You are a professional financial analyst reviewing SEC 10-K filings.
Answer the question using ONLY the context provided below. If the context is insufficient, say you don't have enough information.

Guidelines:
- Write a clear, well-structured answer in plain text
- Use short paragraphs for readability
- Highlight key figures, dates, and metrics when relevant
- Do NOT include chunk_id values or references in your answer — citations are handled separately

Question:
{question}

Context:
{context_block}
"""


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Filing Analyst API", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    """
    Query the RAG system with a natural language question.

    Returns an AI-generated answer with supporting citations from SEC filings.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(req.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")

    store: QdrantStore = request.app.state.store
    embedder: Embedder = request.app.state.embedder
    llm: OllamaLLM = request.app.state.llm

    try:
        qvec = embedder.embed_query(req.query)
    except Exception:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=502, detail="Embedding service unavailable")

    try:
        results = store.search(query_vector=qvec, limit=settings.top_k)
    except Exception:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=502, detail="Vector store unavailable")

    contexts = []
    citations: list[Citation] = []

    for r in results:
        p = r.payload or {}
        text = p.get("text", "")
        chunk_id = p.get("chunk_id")

        contexts.append({
            "chunk_id": chunk_id,
            "text": text[:800],
        })

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

    try:
        prompt = build_prompt(req.query, contexts)
        answer = llm.generate(prompt)
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail="LLM service unavailable")

    return QueryResponse(answer=answer, citations=citations)


@app.get("/stats")
def get_stats(request: Request):
    """Get system statistics"""
    store: QdrantStore = request.app.state.store
    try:
        collection_info = store.client.get_collection(settings.qdrant_collection)
        return {
            "collection_name": settings.qdrant_collection,
            "vector_count": collection_info.vectors_count,
            "indexed_points": collection_info.points_count,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.ollama_model,
        }
    except Exception:
        logger.exception("Failed to retrieve collection stats")
        raise HTTPException(status_code=503, detail="Could not retrieve stats from vector store")


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Filing Analyst API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query - Submit a natural language query",
            "stats": "GET /stats - Get system statistics",
            "health": "GET /health - Health check"
        }
    }
