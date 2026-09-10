"""Prompt templates for the generation pipeline."""

QUERY_ANALYSIS_PROMPT = """Analyze the following question for a knowledge graph + vector RAG system.

Identify:
1. The query type (factual, semantic, relationship, or multi_hop). Use "multi_hop" when answering the question requires following two or more connected relationships across intermediate entities.
2. Entity mentions (names of people, teams, projects, technologies, services, vendors, incidents, policies)
3. Relationship hints (types of relationships mentioned or implied: WORKS_IN, MANAGES, OWNS, MAINTAINS, USES, DEPENDS_ON, SUPPORTED_BY, AFFECTS, GOVERNED_BY, REPORTS_TO)

Return JSON only:
{{
  "query_type": "factual|semantic|relationship|multi_hop",
  "entity_mentions": ["entity1", "entity2"],
  "relationship_hints": ["REL_TYPE1", "REL_TYPE2"]
}}

Question: {question}

JSON output:"""


GENERATION_PROMPT = """Answer the user's question using ONLY the supplied evidence below.

Rules:
- Do not invent facts or relationships not in the evidence.
- If the evidence is insufficient, clearly state that the answer cannot be determined.
- For relationship-heavy questions, use the supplied graph paths.
- Cite evidence using IDs like [V1], [V2] for vector evidence and [G1], [G2] for graph evidence.
- Be concise but thorough.

Evidence:
{evidence}

Question: {question}

Answer:"""


def format_evidence_for_prompt(
    vector_evidence: list,
    graph_evidence: list,
) -> str:
    """Format evidence items for the generation prompt."""
    parts = []
    
    for i, ev in enumerate(vector_evidence):
        citation_id = f"V{i + 1}"
        doc_name = ev.metadata.get("document_name", "unknown") if hasattr(ev, 'metadata') else ev.get("metadata", {}).get("document_name", "unknown")
        text = ev.text if hasattr(ev, 'text') else ev.get("text", "")
        parts.append(f"[{citation_id}]\nSource: {doc_name}\nEvidence:\n{text}\n")
    
    for i, ev in enumerate(graph_evidence):
        citation_id = f"G{i + 1}"
        text = ev.text if hasattr(ev, 'text') else ev.get("text", "")
        doc_ids = ev.source_document_ids if hasattr(ev, 'source_document_ids') else ev.get("source_document_ids", [])
        parts.append(f"[{citation_id}]\nGraph path:\n{text}\nSource documents:\n{', '.join(doc_ids)}\n")
    
    return "\n".join(parts)
