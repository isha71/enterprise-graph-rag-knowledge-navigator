"""FastAPI application entry point."""
from fastapi import FastAPI
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import health, documents, query, graph

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Enterprise GraphRAG Knowledge Navigator - A GenAI knowledge assistant combining vector retrieval with knowledge graph traversal.",
    version="1.0.0",
)

app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(query.router, tags=["Query"])
app.include_router(graph.router, prefix="/graph", tags=["Graph"])

@app.on_event("startup")
async def startup():
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
