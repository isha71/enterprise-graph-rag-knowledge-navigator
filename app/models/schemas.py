from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    neo4j: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_created: int
    vector_points_upserted: int
    entities_extracted: int
    relationships_extracted: int
    graph_chunks_attempted: int = 0
    graph_chunks_succeeded: int = 0
    graph_chunks_failed: int = 0


class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int


class QueryRequest(BaseModel):
    question: str


class RetrieveRequest(BaseModel):
    question: str


class Citation(BaseModel):
    citation_id: str
    type: str
    document_name: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    path: str | None = None
    documents: list[str] | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    graph_evidence: list[dict] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query_analysis: dict = Field(default_factory=dict)
    vector_evidence: list[dict] = Field(default_factory=list)
    graph_evidence: list[dict] = Field(default_factory=list)
    merged_evidence: list[dict] = Field(default_factory=list)
    reranked_evidence: list[dict] = Field(default_factory=list)
