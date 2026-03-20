# RAG Filing Analyst

**Professional-grade AI platform for intelligent SEC filing analysis using Retrieval-Augmented Generation (RAG).**

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd rag-filing-analyst

# 2. Start vector database
docker compose up -d qdrant

# 3. Setup backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.index_sec_dataset
uvicorn src.api.main:app --reload &

# 4. Setup frontend  
cd ../frontend
npm install && npm run dev
```

**Access:** http://localhost:3000

## 🛠️ Development

See detailed setup instructions in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## 📚 Documentation

- **[API Documentation](docs/API.md)** - Complete API reference
- **[Development Guide](docs/DEVELOPMENT.md)** - Setup and development workflow

## ✨ Features

- 🧠 **AI-Powered Analysis** - Natural language queries of SEC filings
- 🔍 **Semantic Search** - Vector-based document retrieval  
- 📊 **Citation Tracking** - Traceable sources with confidence scores
- 🚀 **Modern Stack** - React + TypeScript frontend, FastAPI backend
- 🐳 **Docker Support** - Containerized deployment
- 🔒 **Privacy-First** - Local LLM inference, no external API calls

## 🏗️ Architecture

```
React Frontend ↔ FastAPI Backend ↔ Qdrant Vector DB ↔ Local LLM
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

⭐ **Star this repo if you find it useful!**