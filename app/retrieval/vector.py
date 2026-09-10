"""Vector-only retrieval."""
from app.vectorstore.qdrant_store import QdrantStore
from app.models.domain import EvidenceItem
from app.core.logging import get_logger

logger = get_logger(__name__)


async def retrieve_vector_evidence(
    question: str,
    qdrant_store: QdrantStore,
    top_k: int | None = None,
) -> list[EvidenceItem]:
    """Retrieve evidence from Qdrant vector store."""
    evidence = qdrant_store.search(question, top_k=top_k)
    logger.info(f"Vector evidence items: {len(evidence)}")
    return evidence
