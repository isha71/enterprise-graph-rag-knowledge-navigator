import re

from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse
from app.workflow.graph import run_query
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _sanitize_error(error: str) -> str:
    """Strip raw exception details, keeping only the error category."""
    return re.sub(r":.*", "", error).strip()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute GraphRAG query through LangGraph workflow."""
    try:
        result = await run_query(request.question, mode="graphrag")
        
        raw_errors = result.get("errors", [])
        if raw_errors:
            logger.warning("Workflow errors during query: %s", raw_errors)
        
        return QueryResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            graph_evidence=[
                {"path": p} for p in result.get("graph_paths", [])
            ],
            debug={
                "query_analysis": result.get("query_analysis", {}),
                "vector_evidence_count": len(result.get("vector_evidence", [])),
                "graph_evidence_count": len(result.get("graph_evidence", [])),
                "merged_evidence_count": len(result.get("merged_evidence", [])),
                "reranked_evidence_count": len(result.get("reranked_evidence", [])),
                "errors": [_sanitize_error(e) for e in raw_errors],
            },
        )
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed.")

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Debug retrieval endpoint - returns all retrieval stages."""
    try:
        result = await run_query(request.question, mode="graphrag")
        
        return RetrieveResponse(
            query_analysis=result.get("query_analysis", {}),
            vector_evidence=result.get("vector_evidence", []),
            graph_evidence=result.get("graph_evidence", []),
            merged_evidence=result.get("merged_evidence", []),
            reranked_evidence=result.get("reranked_evidence", []),
        )
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail="Retrieval processing failed.")
