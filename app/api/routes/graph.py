from fastapi import APIRouter, HTTPException
from app.graph.neo4j_store import Neo4jStore
from app.graph.normalization import normalize_entity_name, compute_entity_id
from app.models.domain import EntityType
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/entity/{name}")
async def get_entity(name: str):
    """Get entity and its neighbors from knowledge graph."""
    neo4j = Neo4jStore()
    try:
        entity = await neo4j.find_entity(name)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        
        result = await neo4j.get_entity_neighbors(entity["id"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Graph entity lookup failed")
        raise HTTPException(status_code=500, detail="Graph lookup failed.")
    finally:
        await neo4j.close()
