from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.loaders import load_document
from app.models.domain import DocumentChunk

logger = get_logger(__name__)


def run_ingestion_pipeline(
    file_bytes: bytes,
    filename: str,
) -> tuple[str, list[DocumentChunk]]:
    """
    Full ingestion pipeline: load -> clean -> chunk.

    Returns (document_id, chunks).
    """
    settings = get_settings()

    # 1. Load document
    document_id, sections = load_document(file_bytes, filename)
    logger.info(f"Document loaded: {filename}, id={document_id}")

    # 2. Clean and chunk each section
    all_chunks: list[DocumentChunk] = []
    chunk_index = 0
    for section in sections:
        cleaned = clean_text(section["text"])
        if not cleaned.strip():
            continue
        chunks = chunk_text(
            text=cleaned,
            document_id=document_id,
            document_name=filename,
            page_number=section.get("page_number"),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            start_index=chunk_index,
        )
        all_chunks.extend(chunks)
        chunk_index += len(chunks)

    logger.info(f"Chunks created: {len(all_chunks)}")
    return document_id, all_chunks
