from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    EMPLOYEE = "Employee"
    TEAM = "Team"
    DEPARTMENT = "Department"
    PROJECT = "Project"
    TECHNOLOGY = "Technology"
    SERVICE = "Service"
    VENDOR = "Vendor"
    INCIDENT = "Incident"
    POLICY = "Policy"


class RelationshipType(str, Enum):
    WORKS_IN = "WORKS_IN"
    MANAGES = "MANAGES"
    OWNS = "OWNS"
    MAINTAINS = "MAINTAINS"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    SUPPORTED_BY = "SUPPORTED_BY"
    AFFECTS = "AFFECTS"
    GOVERNED_BY = "GOVERNED_BY"
    REPORTS_TO = "REPORTS_TO"


class QueryType(str, Enum):
    FACTUAL = "factual"
    SEMANTIC = "semantic"
    RELATIONSHIP = "relationship"
    MULTI_HOP = "multi_hop"


class DocumentChunk(BaseModel):
    document_id: str
    document_name: str
    page_number: int | None = None
    chunk_id: str
    chunk_index: int
    text: str


class ExtractedEntity(BaseModel):
    name: str
    type: EntityType


class ExtractedRelationship(BaseModel):
    source: str
    type: RelationshipType
    target: str


class GraphExtraction(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


class GraphNode(BaseModel):
    id: str
    name: str
    normalized_name: str
    type: EntityType
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)


class GraphRelationship(BaseModel):
    source_id: str
    target_id: str
    type: RelationshipType
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)


class GraphPath(BaseModel):
    nodes: list[dict]
    relationships: list[str]
    text: str


class EvidenceItem(BaseModel):
    id: str
    text: str
    evidence_type: Literal["vector", "graph"]
    retrieval_score: float | None = None
    source_document_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class QueryAnalysis(BaseModel):
    query_type: QueryType = QueryType.FACTUAL
    entity_mentions: list[str] = Field(default_factory=list)
    relationship_hints: list[str] = Field(default_factory=list)
