# RAG Filing Analyst

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18-blue.svg?logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

A Retrieval-Augmented Generation (RAG) system for querying and analyzing SEC 10-K filings. Ask questions in natural language and get answers backed by cited source documents — all processed locally with no data sent to external APIs.

![Homepage](docs/screenshots/homepage.png)

![Search Results](docs/screenshots/search-results.png)

## Architecture

```
┌─────────────────────┐
│  React + TypeScript  │  Frontend (Vite, Tailwind CSS)
└─────────┬───────────┘
          │ REST API
┌─────────▼───────────┐
│  FastAPI + Python    │  Backend (Pydantic, uvicorn)
└─────────┬───────────┘
          │
    ┌─────▼─────┐     ┌──────────────┐
    │  Qdrant   │     │  Ollama LLM  │
    │  vectors  │     │  (llama3.1)  │
    └───────────┘     └──────────────┘
```

| Layer | Tech |
|-------|------|
| Frontend | React 18, TypeScript, Tailwind CSS, Vite |
| Backend | FastAPI, Pydantic, Python 3.11+ |
| Vector Store | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | llama3.1:8b via Ollama |
| Data | SEC 10-K filings (Hugging Face) |

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Ollama](https://ollama.ai/) with `llama3.1:8b` pulled

```bash
ollama pull llama3.1:8b
```

### Run with Docker (recommended)

```bash
git clone <repository-url>
cd rag-filing-analyst

# Start all services
docker compose --profile full up -d

# Index SEC filings into the vector database
docker exec rag-api python -m scripts.index_sec_dataset

# Open http://localhost:3000
```

### Run locally

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.index_sec_dataset
uvicorn src.api.main:app --reload --port 8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Available services

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| Qdrant Dashboard | http://localhost:6333/dashboard |

## Usage

### Web Interface

Enter a question in the search bar. The system retrieves relevant filing excerpts via vector similarity search, then generates an answer using the local LLM with citations.

**Example queries:**
- "What are the main risk factors mentioned in recent filings?"
- "Summarize the business model and revenue streams"
- "What regulatory challenges does the company face?"

### API

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key risk factors?"}'

# Health
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/stats
```

## Project Structure

```
rag-filing-analyst/
├── frontend/                  # React TypeScript SPA
│   ├── src/
│   │   ├── components/        # UI components
│   │   ├── hooks/             # Custom React hooks
│   │   ├── services/          # API client (Axios)
│   │   ├── types/             # TypeScript interfaces
│   │   └── utils/             # Helpers
│   ├── Dockerfile             # Production (nginx)
│   └── Dockerfile.dev         # Development (Vite HMR)
├── backend/                   # FastAPI application
│   ├── src/
│   │   ├── api/               # Endpoints + schemas
│   │   └── rag_core/          # RAG pipeline (embeddings, LLM, vectorstore, chunking)
│   ├── scripts/               # Data indexing
│   ├── tests/                 # pytest suite
│   ├── Dockerfile             # Production
│   └── Dockerfile.dev         # Development (hot reload)
├── docs/                      # Documentation
├── .github/workflows/         # CI/CD (lint, test, build, security scan)
├── docker-compose.yml         # Development services
├── docker-compose.prod.yml    # Production deployment
└── Makefile                   # Dev commands
```

## Development

```bash
make help          # Show all available commands
make dev           # Start all services
make test-backend  # Run pytest
make lint-backend  # Run black + isort checks
make lint-frontend # Run ESLint
make index         # Index SEC filings
make logs          # Tail container logs
make down          # Stop services
```

### Manual setup

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                             # Run tests
black --check src/ tests/                          # Format check
uvicorn src.api.main:app --reload --port 8000      # Dev server
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev          # Dev server with HMR
npm run build        # Production build
npm run lint         # ESLint
npm run type-check   # TypeScript validation
npm run format       # Prettier
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + dependency status |
| `POST` | `/query` | Natural language query with RAG |
| `GET` | `/stats` | Collection and model statistics |
| `GET` | `/docs` | Interactive Swagger documentation |

See [docs/API.md](docs/API.md) for full request/response schemas and examples.

## License

[MIT](LICENSE)
