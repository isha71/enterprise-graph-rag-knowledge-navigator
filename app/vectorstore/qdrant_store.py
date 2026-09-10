from __future__ import annotations

import hashlib
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.embeddings.service import EmbeddingService
from app.models.domain import DocumentChunk, EvidenceItem

logger = get_logger(__name__)


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a chunk_id hash to a deterministic UUID string for Qdrant."""
    return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))


class QdrantStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._collection = settings.qdrant_collection
        self._embedding_service = EmbeddingService.get_instance()

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedding_service.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self._collection}")

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index document chunks into Qdrant. Returns number of points upserted."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedding_service.embed(texts)

        points: list[PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = _chunk_id_to_uuid(chunk.chunk_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "page_number": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.info(f"Upserted {len(points)} points to Qdrant")
        return len(points)

    def search(self, query: str, top_k: int | None = None) -> list[EvidenceItem]:
        """Semantic search against Qdrant."""
        settings = get_settings()
        top_k = top_k or settings.vector_top_k

        query_embedding = self._embedding_service.embed_single(query)
        query_vector = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        )

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results = response.points

        evidence: list[EvidenceItem] = []
        for r in results:
            evidence.append(
                EvidenceItem(
                    id=r.payload.get("chunk_id", ""),
                    text=r.payload.get("text", ""),
                    evidence_type="vector",
                    retrieval_score=r.score,
                    source_document_ids=[r.payload.get("document_id", "")],
                    source_chunk_ids=[r.payload.get("chunk_id", "")],
                    metadata={
                        "document_name": r.payload.get("document_name", ""),
                        "page_number": r.payload.get("page_number"),
                        "chunk_index": r.payload.get("chunk_index"),
                    },
                )
            )

        return evidence

    def get_documents(self) -> list[dict]:
        """Get list of indexed documents with chunk counts."""
        all_points: list = []
        offset = None
        while True:
            results, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        docs: dict[str, dict] = {}
        for point in all_points:
            doc_id = point.payload.get("document_id", "")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "document_name": point.payload.get("document_name", ""),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def is_healthy(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
