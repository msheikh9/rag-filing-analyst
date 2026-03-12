#RAG Filing Analyst (Local, Production-Style)

A fully local Retrieval-Augmented Generation (RAG) service that performs grounded question answering over SEC 10-K financial filings.

This system:

Indexes real SEC filings into a vector database (Qdrant)

Uses local sentence-transformer embeddings

Uses a local LLM via Ollama

Returns source-grounded answers with ranked citations

Exposes a production-style FastAPI service with OpenAPI docs

Architecture

User Query
↓
Embed query (Sentence-Transformers)
↓
Vector search (Qdrant)
↓
Retrieve top-k relevant filing chunks
↓
Construct grounded prompt
↓
Generate answer (Ollama LLM)
↓
Return JSON with answer + citations

Tech Stack

Python 3.11

FastAPI

Qdrant (Docker)

Sentence-Transformers (local embeddings)

Ollama (local LLM)

Hugging Face Datasets (SEC filings)

Quickstart
1. Start Qdrant

docker compose up -d

2. Install dependencies

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3. Pull local model

ollama pull llama3.1:8b

4. Index dataset

python3 -m scripts.index_sec_dataset

5. Run API

uvicorn src.api.main:app --reload

Visit:

http://localhost:8000/docs

Example Query

POST /query

{
"query": "Summarize the key risk factors mentioned."
}

Response:

Grounded answer

Ranked citations

Filing date

Document ID

Section metadata

Why RAG?

Large language models alone hallucinate and lack source traceability.

RAG:

Grounds answers in real documents

Improves factual reliability

Provides auditability via citations

Enables domain-specific knowledge systems

Future Improvements

Add hybrid BM25 + vector retrieval

Add reranking stage

Add evaluation metrics (Recall@K)

Support multi-year comparison

Add UI dashboard