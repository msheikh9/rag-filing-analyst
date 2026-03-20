import pytest


pytestmark = pytest.mark.unit


class TestHealthEndpoint:
    def test_returns_healthy_status(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "service" in data


class TestRootEndpoint:
    def test_returns_api_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert "query" in data["endpoints"]
        assert "stats" in data["endpoints"]
        assert "health" in data["endpoints"]

    def test_includes_docs_link(self, client):
        resp = client.get("/")
        assert resp.json()["docs"] == "/docs"


class TestQueryEndpoint:
    def test_valid_query_returns_answer_and_citations(self, client):
        resp = client.post("/query", json={"query": "What is the company revenue?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["citations"], list)
        assert len(data["citations"]) == 2

    def test_citation_fields_populated(self, client):
        resp = client.post("/query", json={"query": "Revenue?"})
        citation = resp.json()["citations"][0]
        assert citation["score"] == pytest.approx(0.95)
        assert citation["chunk_id"] == "chunk_1"
        assert citation["company"] == "Apple Inc."
        assert citation["year"] == "2023"
        assert citation["filingDate"] == "2023-10-30"
        assert citation["docID"] == "0000320193-23-000106"
        assert citation["section"] == "Item 1"
        assert citation["snippet"] is not None

    def test_empty_query_returns_400(self, client):
        resp = client.post("/query", json={"query": "   "})
        assert resp.status_code == 400

    def test_query_too_long_returns_400(self, client):
        resp = client.post("/query", json={"query": "x" * 1001})
        assert resp.status_code == 400

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_calls_embedder_with_input(self, client, mock_embedder):
        client.post("/query", json={"query": "What is net income?"})
        mock_embedder.embed_query.assert_called_once_with("What is net income?")

    def test_query_calls_store_search(self, client, mock_qdrant_store):
        client.post("/query", json={"query": "Revenue?"})
        mock_qdrant_store.search.assert_called_once()
        call_kwargs = mock_qdrant_store.search.call_args
        assert "query_vector" in call_kwargs.kwargs or len(call_kwargs.args) > 0

    def test_query_calls_llm_generate(self, client, mock_llm):
        client.post("/query", json={"query": "Expenses?"})
        mock_llm.generate.assert_called_once()
        prompt = mock_llm.generate.call_args[0][0]
        assert "Expenses?" in prompt

    def test_llm_failure_returns_502(self, client, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("LLM unavailable")
        resp = client.post("/query", json={"query": "Revenue?"})
        assert resp.status_code == 502

    def test_snippet_truncated_to_243_chars(self, client, mock_qdrant_store, search_result_factory):
        long_text = "A" * 500
        mock_qdrant_store.search.return_value = [search_result_factory(text=long_text)]
        resp = client.post("/query", json={"query": "Test?"})
        snippet = resp.json()["citations"][0]["snippet"]
        assert snippet.endswith("...")
        assert len(snippet) == 243

    def test_empty_search_results_returns_empty_citations(self, client, mock_qdrant_store):
        mock_qdrant_store.search.return_value = []
        resp = client.post("/query", json={"query": "Something obscure?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["citations"] == []
        assert "answer" in data


class TestStatsEndpoint:
    def test_returns_collection_stats(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vector_count"] == 1500
        assert data["indexed_points"] == 1500
        assert "collection_name" in data
        assert "embedding_model" in data
        assert "llm_model" in data

    def test_qdrant_failure_returns_503(self, client, mock_qdrant_store):
        mock_qdrant_store.client.get_collection.side_effect = ConnectionError("Qdrant down")
        resp = client.get("/stats")
        assert resp.status_code == 503


class TestBuildPrompt:
    def test_prompt_contains_question_and_context(self):
        from src.api.main import build_prompt
        contexts = [
            {"chunk_id": "c1", "text": "Revenue was $100M."},
            {"chunk_id": "c2", "text": "Net income grew 15%."},
        ]
        prompt = build_prompt("What is the revenue?", contexts)
        assert "What is the revenue?" in prompt
        assert "Revenue was $100M." in prompt
        assert "Net income grew 15%." in prompt
        assert "[chunk_id=c1]" in prompt
        assert "[chunk_id=c2]" in prompt

    def test_prompt_empty_contexts(self):
        from src.api.main import build_prompt
        prompt = build_prompt("Any question?", [])
        assert "Any question?" in prompt
