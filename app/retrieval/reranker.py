"""Cross-encoder reranking."""
from sentence_transformers import CrossEncoder
from app.models.domain import EvidenceItem
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RerankerService:
    _instance = None
    _model = None
    
    @classmethod
    def get_instance(cls) -> "RerankerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        settings = get_settings()
        logger.info(f"Loading reranker model: {settings.reranker_model}")
        self._model = CrossEncoder(settings.reranker_model)
    
    def rerank(
        self, question: str, evidence: list[EvidenceItem], top_k: int | None = None
    ) -> list[EvidenceItem]:
        """Rerank evidence items using cross-encoder."""
        settings = get_settings()
        top_k = top_k or settings.final_context_top_k
        
        if not evidence:
            return []
        
        pairs = [(question, item.text) for item in evidence]
        scores = self._model.predict(pairs)
        
        scored = list(zip(evidence, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for item, score in scored[:top_k]:
            reranked = item.model_copy()
            reranked.retrieval_score = float(score)
            result.append(reranked)
        
        return result
