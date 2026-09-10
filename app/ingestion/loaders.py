from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import UnsupportedFormatError

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def compute_document_id(file_bytes: bytes) -> str:
    """Generate deterministic document ID from file content."""
    return hashlib.sha256(file_bytes).hexdigest()[:24]


def load_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Parse PDF, return list of {'page_number': int, 'text': str}."""
    reader = PdfReader(BytesIO(file_bytes))
    sections: list[dict] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        sections.append({"page_number": page_num, "text": text})
    return sections


def load_text(file_bytes: bytes, filename: str) -> list[dict]:
    """Parse TXT/MD, return list of {'page_number': None, 'text': str}."""
    text = file_bytes.decode("utf-8")
    return [{"page_number": None, "text": text}]


def load_document(file_bytes: bytes, filename: str) -> tuple[str, list[dict]]:
    """
    Load a document, return (document_id, sections).

    Raises UnsupportedFormatError for unknown extensions.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    document_id = compute_document_id(file_bytes)

    if ext == ".pdf":
        sections = load_pdf(file_bytes, filename)
    else:
        sections = load_text(file_bytes, filename)

    return document_id, sections
