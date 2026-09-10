from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.vectorstore.qdrant_store import QdrantStore
from app.graph.neo4j_store import Neo4jStore

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    qdrant_status = "unavailable"
    neo4j_status = "unavailable"
    
    try:
        qdrant = QdrantStore()
        if qdrant.is_healthy():
            qdrant_status = "ok"
    except Exception:
        pass
    
    try:
        neo4j = Neo4jStore()
        try:
            if await neo4j.is_healthy():
                neo4j_status = "ok"
        finally:
            await neo4j.close()
    except Exception:
        pass
    
    overall = "ok" if qdrant_status == "ok" and neo4j_status == "ok" else "degraded"
    
    return HealthResponse(status=overall, qdrant=qdrant_status, neo4j=neo4j_status)
