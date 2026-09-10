from __future__ import annotations

import hashlib

from app.models.domain import DocumentChunk


def compute_chunk_id(
    document_id: str,
    page_number: int | None,
    chunk_index: int,
    text: str,
) -> str:
    """Generate deterministic chunk ID."""
    raw = f"{document_id}:{page_number}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def chunk_text(
    text: str,
    document_id: str,
    document_name: str,
    page_number: int | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 180,
    start_index: int = 0,
) -> list[DocumentChunk]:
    """
    Split text into overlapping chunks.

    Returns list of DocumentChunk with proper metadata.
    """
    if not text:
        return []

    step = chunk_size - chunk_overlap
    if step <= 0:
        step = 1

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = start_index

    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]

        chunk_id = compute_chunk_id(document_id, page_number, chunk_index, chunk_content)

        chunks.append(
            DocumentChunk(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                text=chunk_content,
            )
        )

        chunk_index += 1
        start += step

    return chunks
