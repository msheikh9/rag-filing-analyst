# RAG Filing Analyst - Development Setup

## Quick Development Setup

This guide helps you get the RAG Filing Analyst running in development mode.

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- Docker and Docker Compose
- Ollama (for local LLM)

### 1. Start Services

**Start Qdrant (Vector Database):**
```bash
docker compose up -d qdrant
```

**Install and Start Ollama:**
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull llama3.1:8b
```

### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Index the dataset (this may take a while)
python -m scripts.index_sec_dataset

# Start development server
uvicorn src.api.main:app --reload --port 8000
```

Backend will be available at: http://localhost:8000

### 3. Setup Frontend

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:3000

### 4. Verify Setup

1. Check backend health: http://localhost:8000/health
2. View API docs: http://localhost:8000/docs  
3. Access the app: http://localhost:3000

### Development Commands

**Backend:**
```bash
cd backend
python -m uvicorn src.api.main:app --reload          # Dev server
python -m scripts.index_sec_dataset                  # Re-index data
python -m pytest tests/ -v                          # Run tests
```

**Frontend:**
```bash
cd frontend  
npm run dev          # Dev server
npm run build        # Production build
npm run lint         # Lint code
npm run format       # Format code
npm run type-check   # Check types
```

**Docker Development:**
```bash
# Start all services with Docker
docker compose --profile dev up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Troubleshooting

**Backend Issues:**
- Ensure Qdrant is running: `curl http://localhost:6333/health`
- Verify Ollama is running: `ollama list`
- Check Python environment: `which python`

**Frontend Issues:**
- Clear npm cache: `npm cache clean --force`
- Delete node_modules: `rm -rf node_modules && npm install`
- Check Node version: `node --version`

**Data Issues:**
- Re-run indexing: `python -m scripts.index_sec_dataset`
- Check Qdrant collection: `curl http://localhost:6333/collections`

### Environment Variables

**Backend (.env):**
```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=sec_filings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
TOP_K=5
LOG_LEVEL=INFO
```

**Frontend (.env):**
```env
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE="RAG Filing Analyst"
```