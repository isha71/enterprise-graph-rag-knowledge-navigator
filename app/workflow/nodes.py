"""LangGraph workflow nodes."""
import json
from app.workflow.state import GraphRAGState
from app.generation.llm import get_llm_client
from app.generation.prompts import QUERY_ANALYSIS_PROMPT, GENERATION_PROMPT, format_evidence_for_prompt
from app.generation.citations import build_citations
from app.retrieval.vector import retrieve_vector_evidence
from app.retrieval.fusion import merge_evidence
from app.retrieval.reranker import RerankerService
from app.graph.retrieval import retrieve_graph_evidence
from app.graph.neo4j_store import Neo4jStore
from app.vectorstore.qdrant_store import QdrantStore
from app.models.domain import QueryAnalysis, QueryType, RelationshipType
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_VALID_RELATIONSHIP_TYPES = {rt.value for rt in RelationshipType}

RELATIONSHIP_HINT_ALIASES: dict[str, str] = {
    "SUPPORTS": "SUPPORTED_BY",
    "SUPPORTED_BY": "SUPPORTED_BY",

    "MANAGED_BY": "MANAGES",
    "MANAGE": "MANAGES",
    "MANAGES": "MANAGES",

    "OWNED_BY": "OWNS",
    "OWN": "OWNS",
    "OWNS": "OWNS",

    "MAINTAINED_BY": "MAINTAINS",
    "MAINTAIN": "MAINTAINS",
    "MAINTAINS": "MAINTAINS",

    "DEPENDS": "DEPENDS_ON",
    "DEPEND_ON": "DEPENDS_ON",
    "DEPENDS_ON": "DEPENDS_ON",
}


def _normalize_relationship_hints(raw_hints: list[str]) -> list[str]:
    """Normalize LLM relationship hints to valid Neo4j relationship types.

    Steps per hint: strip → uppercase → replace spaces/hyphens with "_" →
    resolve alias → keep only valid RelationshipType values → deduplicate.
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in raw_hints:
        if not isinstance(raw, str):
            continue
        normalized = raw.strip().upper().replace(" ", "_").replace("-", "_")
        resolved = RELATIONSHIP_HINT_ALIASES.get(normalized, normalized)
        if resolved in _VALID_RELATIONSHIP_TYPES and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _normalize_query_type(raw: str) -> QueryType:
    """Normalize common query type formatting variations to QueryType enum."""
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return QueryType(normalized)
    except ValueError:
        logger.warning(f"Unknown query type '{raw}', falling back to 'factual'")
        return QueryType.FACTUAL


async def analyze_query(state: GraphRAGState) -> dict:
    """Analyze the question to extract entities and relationship hints."""
    question = state["question"]
    errors = list(state.get("errors", []))
    
    try:
        llm = get_llm_client()
        prompt = QUERY_ANALYSIS_PROMPT.format(question=question)
        json_kwargs: dict = {}
        if get_settings().llm_provider.lower() == "groq":
            json_kwargs["max_completion_tokens"] = 200
        result = await llm.generate_json(prompt, temperature=0, **json_kwargs)
        
        raw_type = result.get("query_type", "factual")
        entity_mentions = result.get("entity_mentions", [])
        raw_hints = result.get("relationship_hints", [])
        
        query_type = _normalize_query_type(raw_type)
        normalized_hints = _normalize_relationship_hints(
            raw_hints if isinstance(raw_hints, list) else []
        )
        
        analysis = QueryAnalysis(
            query_type=query_type,
            entity_mentions=entity_mentions if isinstance(entity_mentions, list) else [],
            relationship_hints=normalized_hints,
        )
        logger.info(f"Query analysis: type={analysis.query_type}, entities={analysis.entity_mentions}")
        return {"query_analysis": analysis.model_dump()}
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        errors.append(f"Query analysis error: {str(e)}")
        return {
            "query_analysis": {"query_type": "factual", "entity_mentions": [], "relationship_hints": []},
            "errors": errors,
        }


async def vector_retrieval(state: GraphRAGState) -> dict:
    """Retrieve evidence from Qdrant vector store."""
    question = state["question"]
    errors = list(state.get("errors", []))
    
    try:
        qdrant = QdrantStore()
        evidence = await retrieve_vector_evidence(question, qdrant)
        return {"vector_evidence": [e.model_dump() for e in evidence]}
    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")
        errors.append(f"Vector retrieval error: {str(e)}")
        return {"vector_evidence": [], "errors": errors}


async def graph_retrieval(state: GraphRAGState) -> dict:
    """Retrieve evidence from Neo4j knowledge graph."""
    mode = state.get("mode", "graphrag")
    errors = list(state.get("errors", []))
    
    # Skip graph retrieval in vector-only mode
    if mode == "vector_only":
        logger.info("Skipping graph retrieval (vector_only mode)")
        return {"graph_evidence": []}
    
    query_analysis = state.get("query_analysis", {})
    entity_mentions = query_analysis.get("entity_mentions", [])
    relationship_hints = query_analysis.get("relationship_hints", [])
    query_type = query_analysis.get("query_type", "factual")
    
    try:
        neo4j = Neo4jStore()
        try:
            evidence = await retrieve_graph_evidence(
                entity_mentions,
                neo4j,
                relationship_hints=relationship_hints,
                query_type=query_type,
            )
            return {"graph_evidence": [e.model_dump() for e in evidence]}
        finally:
            await neo4j.close()
    except Exception as e:
        logger.error(f"Graph retrieval failed: {e}")
        errors.append(f"Graph retrieval error: {str(e)}")
        return {"graph_evidence": [], "errors": errors}


async def merge_context(state: GraphRAGState) -> dict:
    """Merge vector and graph evidence."""
    from app.models.domain import EvidenceItem
    
    vector_raw = state.get("vector_evidence", [])
    graph_raw = state.get("graph_evidence", [])
    
    vector_items = [EvidenceItem(**e) if isinstance(e, dict) else e for e in vector_raw]
    graph_items = [EvidenceItem(**e) if isinstance(e, dict) else e for e in graph_raw]
    
    merged = merge_evidence(vector_items, graph_items)
    return {"merged_evidence": [e.model_dump() for e in merged]}


async def rerank_context(state: GraphRAGState) -> dict:
    """Rerank merged evidence using cross-encoder."""
    from app.models.domain import EvidenceItem
    
    question = state["question"]
    merged_raw = state.get("merged_evidence", [])
    errors = list(state.get("errors", []))
    
    if not merged_raw:
        return {"reranked_evidence": []}
    
    try:
        merged_items = [EvidenceItem(**e) if isinstance(e, dict) else e for e in merged_raw]
        reranker = RerankerService.get_instance()
        reranked = reranker.rerank(question, merged_items)
        return {"reranked_evidence": [e.model_dump() for e in reranked]}
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        errors.append(f"Reranking error: {str(e)}")
        return {"reranked_evidence": merged_raw, "errors": errors}


async def generate_answer(state: GraphRAGState) -> dict:
    """Generate grounded answer using LLM."""
    from app.models.domain import EvidenceItem
    
    question = state["question"]
    reranked_raw = state.get("reranked_evidence", [])
    errors = list(state.get("errors", []))
    
    reranked_items = [EvidenceItem(**e) if isinstance(e, dict) else e for e in reranked_raw]
    
    # Separate vector and graph evidence
    vector_ev = [e for e in reranked_items if e.evidence_type == "vector"]
    graph_ev = [e for e in reranked_items if e.evidence_type == "graph"]
    
    if not reranked_items:
        return {
            "answer": "No evidence was retrieved to answer this question.",
            "citations": [],
            "graph_paths": [],
        }
    
    try:
        llm = get_llm_client()
        evidence_text = format_evidence_for_prompt(vector_ev, graph_ev)
        prompt = GENERATION_PROMPT.format(evidence=evidence_text, question=question)
        
        answer = await llm.generate(prompt, temperature=0)
        citations = build_citations(vector_ev, graph_ev)
        
        graph_paths = [e.text for e in graph_ev]
        
        return {
            "answer": answer.strip(),
            "citations": [c.model_dump() for c in citations],
            "graph_paths": graph_paths,
        }
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        errors.append(f"Generation error: {str(e)}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "graph_paths": [],
            "errors": errors,
        }
