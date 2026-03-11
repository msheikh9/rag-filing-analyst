from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class Citation(BaseModel):
    score: float
    chunk_id: str | None = None
    company: str | None = None
    year: str | None = None
    section: str | None = None
    snippet: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]