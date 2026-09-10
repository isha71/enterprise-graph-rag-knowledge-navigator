"""LangGraph state definition."""
from typing import TypedDict
from app.models.domain import EvidenceItem


class GraphRAGState(TypedDict, total=False):
    question: str
    query_analysis: dict
    vector_evidence: list
    graph_evidence: list
    merged_evidence: list
    reranked_evidence: list
    answer: str
    citations: list
    graph_paths: list
    errors: list
    mode: str  # "graphrag" or "vector_only"
