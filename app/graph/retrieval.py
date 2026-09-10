from app.models.domain import EvidenceItem, GraphPath
from app.graph.neo4j_store import Neo4jStore
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

GENERIC_ENTITY_TERMS = {
    "employee",
    "team",
    "department",
    "project",
    "technology",
    "service",
    "vendor",
    "incident",
    "policy",
}


def filter_generic_entity_names(names: list[str]) -> list[str]:
    """Remove generic entity-type nouns from seed names.

    Only exact single-word generic terms are removed (case-insensitive).
    Multi-word names like "AI Platform Team" or "Project Atlas" are kept.
    Falls back to the original list if filtering removes everything.
    """
    filtered = [n for n in names if n.strip().lower() not in GENERIC_ENTITY_TERMS]
    return filtered if filtered else names


def format_graph_path(path: dict) -> GraphPath:
    """Convert raw path dict to GraphPath with readable text.
    
    Preserves actual Neo4j relationship direction. If the relationship
    was stored as A->B but traversal visits B first, renders B <-[REL]- A.
    """
    nodes = path.get("nodes", [])
    rels = path.get("relationships", [])
    
    parts = []
    for i, node in enumerate(nodes):
        parts.append(node.get("name", "?"))
        if i < len(rels):
            rel_type = rels[i].get("type", "?")
            forward = rels[i].get("forward", True)
            if forward:
                parts.append(f"-[{rel_type}]->")
            else:
                parts.append(f"<-[{rel_type}]-")
    
    text = " ".join(parts)
    
    return GraphPath(
        nodes=[{"name": n.get("name", ""), "type": n.get("type", "")} for n in nodes],
        relationships=[r.get("type", "") for r in rels],
        text=text,
    )

def paths_to_evidence(paths: list[dict]) -> list[EvidenceItem]:
    """Convert graph paths to EvidenceItems.

    Provenance is aggregated primarily from relationships in the path.
    Node provenance is used only as a fallback when no relationship
    provenance exists.
    """
    evidence = []
    seen_texts = set()
    
    for i, path in enumerate(paths):
        graph_path = format_graph_path(path)
        
        if graph_path.text in seen_texts:
            continue
        seen_texts.add(graph_path.text)
        
        rel_chunk_ids: set[str] = set()
        rel_doc_ids: set[str] = set()
        rel_doc_names: set[str] = set()
        for rel in path.get("relationships", []):
            rel_chunk_ids.update(rel.get("source_chunk_ids", []))
            rel_doc_ids.update(rel.get("source_document_ids", []))
            rel_doc_names.update(rel.get("source_document_names", []))

        if rel_doc_ids:
            all_chunk_ids = rel_chunk_ids
            all_doc_ids = rel_doc_ids
            all_doc_names = rel_doc_names
        else:
            all_chunk_ids = set()
            all_doc_ids = set()
            all_doc_names = set()
            for node in path.get("nodes", []):
                all_chunk_ids.update(node.get("source_chunk_ids", []))
                all_doc_ids.update(node.get("source_document_ids", []))
                all_doc_names.update(node.get("source_document_names", []))
        
        evidence.append(EvidenceItem(
            id=f"graph_path_{i}",
            text=graph_path.text,
            evidence_type="graph",
            retrieval_score=None,
            source_document_ids=sorted(all_doc_ids),
            source_chunk_ids=sorted(all_chunk_ids),
            metadata={
                "path_nodes": graph_path.nodes,
                "path_relationships": graph_path.relationships,
                "source_document_names": sorted(all_doc_names),
            },
        ))
    
    return evidence

async def retrieve_graph_evidence(
    entity_names: list[str],
    neo4j_store: Neo4jStore,
    relationship_hints: list[str] | None = None,
    query_type: str | None = None,
) -> list[EvidenceItem]:
    """
    Full graph retrieval: find entities, traverse, convert to evidence.
    """
    settings = get_settings()
    
    if not entity_names:
        logger.info("No entity names for graph retrieval")
        return []
    
    filtered_names = filter_generic_entity_names(entity_names)
    logger.info(f"Seed entities after filtering: {filtered_names}")
    
    # Find seed entities
    seed_entities = await neo4j_store.find_entities_by_names(filtered_names)
    
    if not seed_entities:
        logger.info(f"No entities found in graph for: {filtered_names}")
        return []
    
    logger.info(f"Found {len(seed_entities)} seed entities for graph retrieval")
    
    prefer_longer = (
        query_type == "multi_hop"
        or (hasattr(query_type, "value") and query_type.value == "multi_hop")
    )
    
    # Multi-hop traversal from each seed
    all_paths = []
    for entity in seed_entities:
        paths = await neo4j_store.multi_hop_traversal(
            seed_id=entity["id"],
            max_hops=settings.graph_max_hops,
            max_paths=settings.graph_max_paths,
            relationship_hints=relationship_hints,
            prefer_longer=prefer_longer,
        )
        all_paths.extend(paths)
    
    # Convert to evidence
    evidence = paths_to_evidence(all_paths)
    logger.info(f"Graph evidence items: {len(evidence)}")
    
    return evidence
