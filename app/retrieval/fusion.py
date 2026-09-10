"""Evidence fusion - combining vector and graph evidence."""
from app.models.domain import EvidenceItem
from app.core.logging import get_logger

logger = get_logger(__name__)


def merge_evidence(
    vector_evidence: list[EvidenceItem],
    graph_evidence: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Merge vector and graph evidence, removing exact duplicates."""
    merged = []
    seen_texts = set()
    
    for item in vector_evidence:
        text_key = item.text.strip().lower()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            merged.append(item)
    
    for item in graph_evidence:
        text_key = item.text.strip().lower()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            merged.append(item)
    
    logger.info(f"Merged evidence: {len(vector_evidence)} vector + {len(graph_evidence)} graph = {len(merged)} unique")
    return merged
