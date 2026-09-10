"""Citation generation from evidence."""
from app.models.schemas import Citation


def build_citations(vector_evidence: list, graph_evidence: list) -> list[Citation]:
    """Build citation objects from evidence items."""
    citations = []
    
    for i, ev in enumerate(vector_evidence):
        meta = ev.metadata if hasattr(ev, 'metadata') else ev.get("metadata", {})
        citations.append(Citation(
            citation_id=f"V{i + 1}",
            type="vector",
            document_name=meta.get("document_name"),
            page_number=meta.get("page_number"),
            chunk_id=ev.id if hasattr(ev, 'id') else ev.get("id"),
        ))
    
    for i, ev in enumerate(graph_evidence):
        text = ev.text if hasattr(ev, 'text') else ev.get("text", "")
        meta = ev.metadata if hasattr(ev, 'metadata') else ev.get("metadata", {})
        doc_names = meta.get("source_document_names", [])
        doc_ids = ev.source_document_ids if hasattr(ev, 'source_document_ids') else ev.get("source_document_ids", [])
        citations.append(Citation(
            citation_id=f"G{i + 1}",
            type="graph",
            path=text,
            documents=doc_names if doc_names else doc_ids,
        ))
    
    return citations
