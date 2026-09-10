from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Lightweight deterministic text cleaning."""
    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Collapse repeated whitespace on same line (not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # 5. Strip leading/trailing whitespace
    text = text.strip()

    return text
