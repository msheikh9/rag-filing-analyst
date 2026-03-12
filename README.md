RAG Filing Analyst

A production-style Retrieval-Augmented Generation (RAG) service for grounded question answering over SEC 10-K financial filings.

This project demonstrates an end-to-end RAG pipeline integrating vector search, transformer-based embeddings, and a local large language model to return traceable, citation-backed responses.

Overview

Large language models alone are not reliable knowledge systems.
They lack access to structured domain data and can produce hallucinated outputs.

This system implements Retrieval-Augmented Generation (RAG) to:

Index real SEC 10-K filings into a vector database

Retrieve semantically relevant document chunks

Construct a grounded prompt using retrieved context

Generate an answer constrained to source material

Return ranked citations with metadata for traceability

The result is a reproducible, domain-specific question answering service.

Architecture

User Query
↓
Embedding Model (Sentence-Transformers)
↓
Vector Search (Qdrant)
↓
Top-K Relevant Filing Chunks
↓
Grounded Prompt Construction
↓
Local LLM Inference (Ollama)
↓
Structured JSON Response (Answer + Citations)

Tech Stack

Backend:

Python 3.11

FastAPI

Pydantic

Vector Store:

Qdrant (Dockerized)

Embeddings:

sentence-transformers/all-MiniLM-L6-v2

LLM:

llama3.1:8b (via Ollama)

Dataset:

SEC 10-K filings (Hugging Face dataset)

API
Health Check

GET /health

Query

POST /query

Request:

{
"query": "Summarize the key risk factors mentioned."
}

Response:

{
"answer": "...",
"citations": [
{
"score": 0.67,
"company": "0000001750",
"year": "2020",
"filingDate": "2020-07-21",
"docID": "0000001750_10-K_2020",
"section": "1"
}
]
}

Swagger UI available at:

http://localhost:8000/docs

Quickstart

Start Qdrant

docker compose up -d

Install dependencies

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Pull local model

ollama pull llama3.1:8b

Index dataset

python3 -m scripts.index_sec_dataset

Run API

uvicorn src.api.main:app --reload

Key Capabilities

Semantic retrieval over financial filings

Context-aware answer generation

Ranked citations with document metadata

Modular architecture (vectorstore, embeddings, LLM separated)

Fully local inference pipeline

Reproducible environment

Future Extensions

Hybrid retrieval (BM25 + vector)

Cross-encoder reranking

Retrieval evaluation metrics (Recall@K)

Multi-year comparative analysis

Frontend interface