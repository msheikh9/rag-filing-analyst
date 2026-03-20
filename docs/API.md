# API Documentation

## Overview

The RAG Filing Analyst API provides endpoints for querying SEC filings using natural language processing and retrieval-augmented generation.

**Base URL:** `http://localhost:8000`
**Interactive Docs:** `http://localhost:8000/docs`

## Authentication

Currently, the API does not require authentication. In production deployments, consider adding API key authentication.

## Endpoints

### Health Check

**GET** `/health`

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "service": "RAG Filing Analyst API",
  "version": "1.0.0"
}
```

### Submit Query

**POST** `/query`

Submit a natural language query to analyze SEC filings.

**Request Body:**
```json
{
  "query": "What are the main risk factors mentioned in the filings?"
}
```

**Request Schema:**
- `query` (string, required): Natural language question (max 1000 characters)

**Response:**
```json
{
  "answer": "Based on the SEC filings, the main risk factors include...",
  "citations": [
    {
      "score": 0.85,
      "chunk_id": "chunk_12345",
      "company": "0000001750",
      "year": "2023",
      "filingDate": "2023-03-15",
      "docID": "0000001750_10-K_2023",
      "section": "1A",
      "snippet": "The company faces risks related to..."
    }
  ]
}
```

**Response Schema:**
- `answer` (string): AI-generated response based on filing content
- `citations` (array): Supporting evidence from filings
  - `score` (number): Relevance score (0.0 - 1.0)
  - `chunk_id` (string, optional): Unique chunk identifier
  - `company` (string, optional): Company CIK identifier
  - `year` (string, optional): Filing year
  - `filingDate` (string, optional): Filing date (YYYY-MM-DD)
  - `docID` (string, optional): Document identifier
  - `section` (string, optional): Document section
  - `snippet` (string, optional): Text excerpt (240 chars)

### System Statistics

**GET** `/stats`

Get system statistics and configuration information.

**Response:**
```json
{
  "collection_name": "sec_filings",
  "vector_count": 125000,
  "indexed_points": 125000,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "llama3.1:8b"
}
```

### API Information

**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
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
```

## Error Responses

The API uses standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid query)
- `422` - Validation Error
- `500` - Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error description"
}
```

**Common Errors:**
- `Query cannot be empty` (400)
- `Query too long (max 1000 characters)` (400)
- `Internal server error: {details}` (500)

## Rate Limits

Currently, no rate limits are enforced. For production deployments, consider implementing rate limiting based on your requirements.

## Examples

### cURL Examples

**Health Check:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Submit Query:**
```bash
curl -X POST "http://localhost:8000/query" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "What are the key risk factors mentioned?"}'
```

**Get Statistics:**
```bash
curl -X GET "http://localhost:8000/stats"
```

### Python Example

```python
import requests

# Submit a query
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What are the main business segments?"}
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Citations: {len(data['citations'])}")
```

### JavaScript Example

```javascript
// Submit a query
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'What regulatory challenges are mentioned?'
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
console.log('Citations:', data.citations.length);
```

## Data Sources

The API analyzes SEC 10-K filings from the Hugging Face `sec-filings` dataset, which includes:

- Company annual reports (Form 10-K)
- Filing dates from recent years
- Various sections including business overview, risk factors, financial data
- Metadata including company identifiers, sections, and filing dates

## Performance Considerations

- **Query Latency:** Typically 2-5 seconds for full RAG pipeline
- **Vector Search:** Sub-second retrieval from indexed documents
- **LLM Generation:** 1-3 seconds depending on response length
- **Concurrent Requests:** Backend supports multiple concurrent queries

## Interactive Documentation

Visit `http://localhost:8000/docs` for:
- Interactive API explorer
- Request/response examples
- Schema validation
- Test interface for all endpoints