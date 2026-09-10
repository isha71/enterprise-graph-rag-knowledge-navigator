import re
import unicodedata
import hashlib
from app.models.domain import EntityType

REMOVABLE_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc"}

def normalize_entity_name(name: str, entity_type: EntityType) -> str:
    """Normalize entity name for deduplication."""
    # 1. Unicode NFC normalization
    # 2. Lowercase
    # 3. Remove punctuation except hyphens
    # 4. Collapse whitespace
    # 5. Strip
    # 6. For Vendor type, remove common suffixes
    normalized = unicodedata.normalize("NFC", name)
    normalized = normalized.lower()
    normalized = re.sub(r"[^\w\s\-]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    if entity_type in (EntityType.VENDOR,):
        words = normalized.split()
        while words and words[-1] in REMOVABLE_SUFFIXES:
            words.pop()
        normalized = " ".join(words)
    
    return normalized

def compute_entity_id(entity_type: EntityType, normalized_name: str) -> str:
    """Generate deterministic entity ID."""
    raw = f"{entity_type.value}:{normalized_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
