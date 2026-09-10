from __future__ import annotations


class GraphRAGError(Exception):
    """Base exception for the Enterprise GraphRAG Knowledge Navigator."""


class DocumentParsingError(GraphRAGError):
    """Raised when a document cannot be parsed."""


class UnsupportedFormatError(GraphRAGError):
    """Raised when an unsupported document format is encountered."""


class ExtractionError(GraphRAGError):
    """Raised when entity/relationship extraction fails."""


class GraphQueryError(GraphRAGError):
    """Raised when a graph database query fails."""


class VectorStoreError(GraphRAGError):
    """Raised when a vector store operation fails."""


class LLMError(GraphRAGError):
    """Raised when an LLM call fails."""


class GraphIngestionError(GraphRAGError):
    """Raised when graph ingestion fails completely for a document."""
