import pytest
from pydantic import ValidationError
from src.api.schemas import Citation, QueryRequest, QueryResponse


pytestmark = pytest.mark.unit


class TestQueryRequest:
    def test_valid_query(self):
        req = QueryRequest(query="What is the revenue?")
        assert req.query == "What is the revenue?"

    def test_empty_string_is_accepted(self):
        req = QueryRequest(query="")
        assert req.query == ""

    def test_missing_query_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryRequest()

    def test_non_coercible_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryRequest(query={"nested": "object"})

    def test_extra_fields_ignored(self):
        req = QueryRequest(query="test", extra="should be ignored")
        assert not hasattr(req, "extra") or req.model_config.get("extra") != "forbid"


class TestCitation:
    def test_minimal_citation(self):
        c = Citation(score=0.9)
        assert c.score == pytest.approx(0.9)
        assert c.chunk_id is None
        assert c.company is None
        assert c.year is None
        assert c.filingDate is None
        assert c.docID is None
        assert c.section is None
        assert c.snippet is None

    def test_fully_populated_citation(self):
        c = Citation(
            score=0.95,
            chunk_id="chunk_42",
            company="Tesla Inc.",
            year="2024",
            filingDate="2024-02-15",
            docID="doc-abc-123",
            section="Item 7",
            snippet="Revenue increased by 20%...",
        )
        assert c.chunk_id == "chunk_42"
        assert c.company == "Tesla Inc."
        assert c.year == "2024"
        assert c.filingDate == "2024-02-15"
        assert c.docID == "doc-abc-123"
        assert c.section == "Item 7"
        assert c.snippet == "Revenue increased by 20%..."

    def test_missing_score_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Citation()

    def test_score_must_be_numeric(self):
        with pytest.raises(ValidationError):
            Citation(score="not_a_number")

    def test_score_accepts_integer(self):
        c = Citation(score=1)
        assert c.score == pytest.approx(1.0)


class TestQueryResponse:
    def test_valid_response(self):
        resp = QueryResponse(
            answer="The revenue was $100M.",
            citations=[Citation(score=0.9, chunk_id="c1")],
        )
        assert resp.answer == "The revenue was $100M."
        assert len(resp.citations) == 1

    def test_empty_citations_list(self):
        resp = QueryResponse(answer="No results.", citations=[])
        assert resp.citations == []

    def test_missing_answer_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryResponse(citations=[])

    def test_missing_citations_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryResponse(answer="Some answer")

    def test_multiple_citations(self):
        citations = [Citation(score=0.9 - i * 0.1, chunk_id=f"c{i}") for i in range(5)]
        resp = QueryResponse(answer="Answer.", citations=citations)
        assert len(resp.citations) == 5

    def test_serialization_round_trip(self):
        original = QueryResponse(
            answer="Test answer.",
            citations=[
                Citation(score=0.85, chunk_id="c1", company="Apple", year="2023"),
                Citation(score=0.72, chunk_id="c2", section="Item 1A"),
            ],
        )
        data = original.model_dump()
        restored = QueryResponse(**data)
        assert restored.answer == original.answer
        assert len(restored.citations) == len(original.citations)
        assert restored.citations[0].company == "Apple"
        assert restored.citations[1].section == "Item 1A"
