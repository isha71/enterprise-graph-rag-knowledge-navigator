from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import DocumentUploadResponse, DocumentInfo
from app.ingestion.pipeline import run_ingestion_pipeline
from app.vectorstore.qdrant_store import QdrantStore
from app.graph.neo4j_store import Neo4jStore
from app.graph.extraction import extract_graph_from_text
from app.generation.llm import get_llm_client
from app.core.logging import get_logger
from app.core.exceptions import UnsupportedFormatError, ExtractionError

logger = get_logger(__name__)
router = APIRouter()

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document."""
    file_bytes = await file.read()
    filename = file.filename or "unknown"
    
    try:
        document_id, chunks = run_ingestion_pipeline(file_bytes, filename)
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail="Document ingestion failed.")
    
    # Vector indexing
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    points_upserted = qdrant.index_chunks(chunks)
    
    # Graph extraction — fail the request if infrastructure is unavailable
    try:
        llm = get_llm_client()
    except Exception:
        logger.exception("LLM client initialization failed")
        raise HTTPException(
            status_code=503,
            detail="Graph extraction unavailable: LLM provider failed to initialize.",
        )
    
    try:
        neo4j = Neo4jStore()
        if not await neo4j.is_healthy():
            await neo4j.close()
            raise HTTPException(
                status_code=503,
                detail="Graph extraction unavailable: Neo4j is not reachable.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Neo4j connection failed")
        raise HTTPException(
            status_code=503,
            detail="Graph extraction unavailable: Neo4j connection failed.",
        )
    
    total_entities = 0
    total_relationships = 0
    graph_chunks_attempted = 0
    graph_chunks_succeeded = 0
    graph_chunks_failed = 0
    
    try:
        await neo4j.initialize()
        
        for chunk in chunks:
            graph_chunks_attempted += 1
            try:
                extraction = await extract_graph_from_text(chunk.text, llm)
                e_count, r_count = await neo4j.persist_extraction(
                    extraction, document_id, chunk.chunk_id, filename
                )
                total_entities += e_count
                total_relationships += r_count
                graph_chunks_succeeded += 1
            except ExtractionError as e:
                graph_chunks_failed += 1
                logger.error(f"Graph extraction failed for chunk {chunk.chunk_id}: {e}")
            except Exception:
                logger.exception(f"Graph persistence failed for chunk {chunk.chunk_id}")
                raise
    finally:
        await neo4j.close()
    
    if graph_chunks_attempted > 0 and graph_chunks_succeeded == 0:
        raise HTTPException(
            status_code=500,
            detail="Graph ingestion failed: extraction failed for all chunks.",
        )
    
    return DocumentUploadResponse(
        document_id=document_id,
        document_name=filename,
        chunks_created=len(chunks),
        vector_points_upserted=points_upserted,
        entities_extracted=total_entities,
        relationships_extracted=total_relationships,
        graph_chunks_attempted=graph_chunks_attempted,
        graph_chunks_succeeded=graph_chunks_succeeded,
        graph_chunks_failed=graph_chunks_failed,
    )

@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List indexed documents."""
    try:
        qdrant = QdrantStore()
        docs = qdrant.get_documents()
        return [DocumentInfo(**d) for d in docs]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")
