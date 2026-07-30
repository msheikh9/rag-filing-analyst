import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.rag_core.config import settings
from src.rag_core.llm import OllamaLLM
from src.rag_core.retriever import Retriever

from .schemas import Citation, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG dependencies...")
    app.state.retriever = Retriever(settings)
    app.state.llm = OllamaLLM(settings.ollama_url, settings.ollama_model)
    logger.info(
        "RAG dependencies ready (mode=%s, rerank=%s).",
        settings.retrieval_mode,
        settings.enable_rerank,
    )
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
    context_block = "\n\n".join([f"[chunk_id={c['chunk_id']}] {c['text']}" for c in contexts])
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

    retriever: Retriever = request.app.state.retriever
    llm: OllamaLLM = request.app.state.llm

    try:
        results = retriever.retrieve(req.query, top_k=settings.top_k)
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail="Retrieval unavailable")

    contexts = []
    citations: list[Citation] = []

    for r in results:
        text = r.get("text", "")
        chunk_id = r.get("chunk_id")

        contexts.append(
            {
                "chunk_id": chunk_id,
                "text": text[:800],
            }
        )

        citations.append(
            Citation(
                score=float(r["score"]),
                chunk_id=chunk_id,
                company=r.get("company"),
                year=r.get("year"),
                filingDate=r.get("filingDate"),
                docID=r.get("docID"),
                section=r.get("section"),
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
    retriever: Retriever = request.app.state.retriever
    try:
        collection_info = retriever.store.client.get_collection(settings.qdrant_collection)
        return {
            "collection_name": settings.qdrant_collection,
            "vector_count": collection_info.vectors_count,
            "indexed_points": collection_info.points_count,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.ollama_model,
            "retrieval_mode": settings.retrieval_mode,
            "rerank_enabled": settings.enable_rerank,
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
            "health": "GET /health - Health check",
        },
    }
