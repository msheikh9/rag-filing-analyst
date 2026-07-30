import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


class Settings(BaseModel):
    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "sec_10k_chunks")
    top_k: int = int(os.getenv("TOP_K", "8"))

    # Local embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Retrieval. "hybrid" runs dense + BM25 and fuses with RRF inside Qdrant; "dense" is the
    # original single-vector path, kept so the pre-hybrid baseline stays runnable.
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")
    sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    prefetch_limit: int = int(os.getenv("PREFETCH_LIMIT", "50"))

    # Cross-encoder reranking. Off by default: on the current corpus it costs ~200ms at p50
    # and did not improve nDCG@10 beyond run-to-run variance. See docs/EVALUATION.md.
    enable_rerank: bool = _flag("ENABLE_RERANK")
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_depth: int = int(os.getenv("RERANK_DEPTH", "30"))

    # Ollama (local LLM)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


settings = Settings()
