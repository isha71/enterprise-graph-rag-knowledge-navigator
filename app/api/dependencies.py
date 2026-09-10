"""FastAPI dependencies for service instances."""
from functools import lru_cache
from app.vectorstore.qdrant_store import QdrantStore
from app.graph.neo4j_store import Neo4jStore


@lru_cache()
def get_qdrant_store() -> QdrantStore:
    store = QdrantStore()
    store.ensure_collection()
    return store


def get_neo4j_store() -> Neo4jStore:
    return Neo4jStore()
