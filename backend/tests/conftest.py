from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _make_search_result(text="Sample chunk text", score=0.95, chunk_id="chunk_1",
                        company="Apple Inc.", year="2023", filing_date="2023-10-30",
                        doc_id="0000320193-23-000106", section="Item 1"):
    result = MagicMock()
    result.score = score
    result.payload = {
        "text": text,
        "chunk_id": chunk_id,
        "company": company,
        "year": year,
        "filingDate": filing_date,
        "docID": doc_id,
        "section": section,
    }
    return result


@pytest.fixture()
def search_result_factory():
    return _make_search_result


@pytest.fixture()
def mock_qdrant_store():
    store = MagicMock()
    store.search.return_value = [
        _make_search_result(),
        _make_search_result(chunk_id="chunk_2", score=0.88),
    ]
    collection_info = SimpleNamespace(vectors_count=1500, points_count=1500)
    store.client.get_collection.return_value = collection_info
    return store


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384
    return embedder


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = "Based on the SEC filing, the company reported strong revenue growth."
    return llm


@pytest.fixture()
def client(mock_qdrant_store, mock_embedder, mock_llm):
    from src.api.main import app

    @asynccontextmanager
    async def _test_lifespan(a):
        a.state.store = mock_qdrant_store
        a.state.embedder = mock_embedder
        a.state.llm = mock_llm
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _test_lifespan

    with TestClient(app) as tc:
        yield tc

    app.router.lifespan_context = original_lifespan
