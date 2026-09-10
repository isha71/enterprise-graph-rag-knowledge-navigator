"""Ingest sample documents into the system."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ingestion.pipeline import run_ingestion_pipeline
from app.vectorstore.qdrant_store import QdrantStore
from app.graph.extraction import extract_graph_from_text
from app.graph.neo4j_store import Neo4jStore
from app.generation.llm import get_llm_client
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import ExtractionError, GraphIngestionError

setup_logging("INFO")
logger = get_logger(__name__)

SAMPLE_DOCS_DIR = project_root / "data" / "sample_docs"


async def main():
    if not SAMPLE_DOCS_DIR.exists():
        print(f"Sample docs directory not found: {SAMPLE_DOCS_DIR}")
        sys.exit(1)
    
    files = sorted(SAMPLE_DOCS_DIR.glob("*.md"))
    if not files:
        print("No markdown files found in sample_docs/")
        sys.exit(1)
    
    print(f"Found {len(files)} sample documents")
    print("=" * 60)
    
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    
    neo4j = Neo4jStore()
    await neo4j.initialize()
    
    settings = get_settings()
    llm = get_llm_client()
    groq_delay = settings.groq_request_delay_seconds if settings.llm_provider == "groq" else 0
    
    total_chunks = 0
    total_points = 0
    total_entities = 0
    total_relationships = 0
    total_graph_attempted = 0
    total_graph_succeeded = 0
    total_graph_failed = 0
    
    try:
        for filepath in files:
            print(f"\nIngesting: {filepath.name}")
            file_bytes = filepath.read_bytes()
            
            # Run ingestion pipeline
            document_id, chunks = run_ingestion_pipeline(file_bytes, filepath.name)
            total_chunks += len(chunks)
            print(f"  Chunks created: {len(chunks)}")
            
            # Vector indexing
            points = qdrant.index_chunks(chunks)
            total_points += points
            print(f"  Vector points upserted: {points}")
            
            # Graph extraction
            doc_entities = 0
            doc_rels = 0
            doc_attempted = 0
            doc_succeeded = 0
            doc_failed = 0
            for i, chunk in enumerate(chunks):
                doc_attempted += 1
                if groq_delay and i > 0:
                    await asyncio.sleep(groq_delay)
                try:
                    extraction = await extract_graph_from_text(chunk.text, llm)
                    e_count, r_count = await neo4j.persist_extraction(
                        extraction, document_id, chunk.chunk_id, filepath.name
                    )
                    doc_entities += e_count
                    doc_rels += r_count
                    doc_succeeded += 1
                except ExtractionError as e:
                    doc_failed += 1
                    logger.error(f"  Extraction failed for chunk: {e}")
            
            print(f"  Graph chunks: {doc_succeeded} ok, {doc_failed} failed out of {doc_attempted}")
            
            total_graph_attempted += doc_attempted
            total_graph_succeeded += doc_succeeded
            total_graph_failed += doc_failed
            
            if doc_attempted > 0 and doc_succeeded == 0:
                raise GraphIngestionError(
                    f"Graph extraction failed for all {doc_attempted} chunks "
                    f"in document '{filepath.name}'"
                )
            
            total_entities += doc_entities
            total_relationships += doc_rels
            print(f"  Entities extracted: {doc_entities}")
            print(f"  Relationships extracted: {doc_rels}")
    finally:
        await neo4j.close()
    
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Documents processed: {len(files)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total vector points: {total_points}")
    print(f"Total entities extracted: {total_entities}")
    print(f"Total relationships extracted: {total_relationships}")
    print(f"Graph chunks attempted: {total_graph_attempted}")
    print(f"Graph chunks succeeded: {total_graph_succeeded}")
    print(f"Graph chunks failed:    {total_graph_failed}")


if __name__ == "__main__":
    asyncio.run(main())
