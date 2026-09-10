import pytest

from app.graph.extraction import validate_extraction
from app.graph.schema import (
    is_valid_entity_type,
    is_valid_relationship_type,
    is_valid_relationship_signature,
)
from app.models.domain import EntityType, RelationshipType


def test_valid_entity_type():
    assert is_valid_entity_type("Employee")
    assert is_valid_entity_type("Project")


def test_invalid_entity_type():
    assert not is_valid_entity_type("Building")
    assert not is_valid_entity_type("")


def test_valid_relationship_type():
    assert is_valid_relationship_type("MANAGES")
    assert is_valid_relationship_type("USES")


def test_invalid_relationship_type():
    assert not is_valid_relationship_type("LIKES")
    assert not is_valid_relationship_type("")


def test_validate_valid_extraction():
    raw = {
        "entities": [
            {"name": "Bob Chen", "type": "Employee"},
            {"name": "AI Platform Team", "type": "Team"},
        ],
        "relationships": [
            {"source": "Bob Chen", "type": "MANAGES", "target": "AI Platform Team"},
        ],
    }
    result = validate_extraction(raw)
    assert len(result.entities) == 2
    assert len(result.relationships) == 1


def test_reject_invalid_entity_type():
    raw = {
        "entities": [
            {"name": "Bob", "type": "Person"},  # Invalid type
            {"name": "AI Team", "type": "Team"},
        ],
        "relationships": [],
    }
    result = validate_extraction(raw)
    assert len(result.entities) == 1
    assert result.entities[0].name == "AI Team"


def test_reject_invalid_relationship_type():
    raw = {
        "entities": [
            {"name": "Bob", "type": "Employee"},
            {"name": "Team", "type": "Team"},
        ],
        "relationships": [
            {"source": "Bob", "type": "LIKES", "target": "Team"},  # Invalid
        ],
    }
    result = validate_extraction(raw)
    assert len(result.relationships) == 0


def test_reject_relationship_missing_entity():
    raw = {
        "entities": [
            {"name": "Bob", "type": "Employee"},
        ],
        "relationships": [
            {"source": "Bob", "type": "MANAGES", "target": "Unknown Team"},
        ],
    }
    result = validate_extraction(raw)
    assert len(result.relationships) == 0


def test_deduplicate_entities():
    raw = {
        "entities": [
            {"name": "Bob", "type": "Employee"},
            {"name": "Bob", "type": "Employee"},  # Duplicate
        ],
        "relationships": [],
    }
    result = validate_extraction(raw)
    assert len(result.entities) == 1


def test_empty_name_rejected():
    raw = {
        "entities": [
            {"name": "", "type": "Employee"},
            {"name": "Bob", "type": "Employee"},
        ],
        "relationships": [],
    }
    result = validate_extraction(raw)
    assert len(result.entities) == 1


# ---- Relationship signature validation (is_valid_relationship_signature) ----

_ACCEPTED_SIGNATURES = [
    (RelationshipType.WORKS_IN, EntityType.EMPLOYEE, EntityType.TEAM),
    (RelationshipType.MANAGES, EntityType.EMPLOYEE, EntityType.TEAM),
    (RelationshipType.OWNS, EntityType.TEAM, EntityType.PROJECT),
    (RelationshipType.MAINTAINS, EntityType.EMPLOYEE, EntityType.PROJECT),
    (RelationshipType.MAINTAINS, EntityType.TEAM, EntityType.SERVICE),
    (RelationshipType.USES, EntityType.PROJECT, EntityType.TECHNOLOGY),
    (RelationshipType.USES, EntityType.SERVICE, EntityType.TECHNOLOGY),
    (RelationshipType.DEPENDS_ON, EntityType.PROJECT, EntityType.SERVICE),
    (RelationshipType.DEPENDS_ON, EntityType.SERVICE, EntityType.SERVICE),
    (RelationshipType.SUPPORTED_BY, EntityType.TECHNOLOGY, EntityType.VENDOR),
    (RelationshipType.AFFECTS, EntityType.INCIDENT, EntityType.SERVICE),
    (RelationshipType.GOVERNED_BY, EntityType.PROJECT, EntityType.POLICY),
    (RelationshipType.REPORTS_TO, EntityType.EMPLOYEE, EntityType.DEPARTMENT),
]


@pytest.mark.parametrize("rel,src,tgt", _ACCEPTED_SIGNATURES)
def test_accepted_relationship_signature(rel, src, tgt):
    assert is_valid_relationship_signature(rel, src, tgt) is True


_REJECTED_SIGNATURES = [
    (RelationshipType.USES, EntityType.TEAM, EntityType.TECHNOLOGY),
    (RelationshipType.USES, EntityType.TEAM, EntityType.PROJECT),
    (RelationshipType.DEPENDS_ON, EntityType.PROJECT, EntityType.TECHNOLOGY),
    (RelationshipType.WORKS_IN, EntityType.TEAM, EntityType.DEPARTMENT),
    (RelationshipType.USES, EntityType.EMPLOYEE, EntityType.TECHNOLOGY),
    (RelationshipType.SUPPORTED_BY, EntityType.TECHNOLOGY, EntityType.PROJECT),
    (RelationshipType.GOVERNED_BY, EntityType.SERVICE, EntityType.POLICY),
]


@pytest.mark.parametrize("rel,src,tgt", _REJECTED_SIGNATURES)
def test_rejected_relationship_signature(rel, src, tgt):
    assert is_valid_relationship_signature(rel, src, tgt) is False


# ---- validate_extraction with mixed valid/invalid signatures ----

def test_validate_extraction_mixed_signatures():
    """Only relationships with valid signatures survive validation."""
    raw = {
        "entities": [
            {"name": "Bob Chen", "type": "Employee"},
            {"name": "AI Platform Team", "type": "Team"},
            {"name": "Project Atlas", "type": "Project"},
            {"name": "Neo4j", "type": "Technology"},
        ],
        "relationships": [
            # Valid: Employee -MANAGES-> Team
            {"source": "Bob Chen", "type": "MANAGES", "target": "AI Platform Team"},
            # Valid: Team -OWNS-> Project
            {"source": "AI Platform Team", "type": "OWNS", "target": "Project Atlas"},
            # Valid: Project -USES-> Technology
            {"source": "Project Atlas", "type": "USES", "target": "Neo4j"},
            # INVALID: Team -USES-> Technology (Team not allowed for USES)
            {"source": "AI Platform Team", "type": "USES", "target": "Neo4j"},
            # INVALID: Project -DEPENDS_ON-> Technology (must target Service)
            {"source": "Project Atlas", "type": "DEPENDS_ON", "target": "Neo4j"},
        ],
    }
    result = validate_extraction(raw)
    assert len(result.entities) == 4
    assert len(result.relationships) == 3
    rel_tuples = [(r.source, r.type.value, r.target) for r in result.relationships]
    assert ("Bob Chen", "MANAGES", "AI Platform Team") in rel_tuples
    assert ("AI Platform Team", "OWNS", "Project Atlas") in rel_tuples
    assert ("Project Atlas", "USES", "Neo4j") in rel_tuples


def test_entities_retained_when_relationship_rejected():
    """Entities survive even if all their relationships are rejected."""
    raw = {
        "entities": [
            {"name": "AI Platform Team", "type": "Team"},
            {"name": "Neo4j", "type": "Technology"},
        ],
        "relationships": [
            # INVALID: Team -USES-> Technology
            {"source": "AI Platform Team", "type": "USES", "target": "Neo4j"},
        ],
    }
    result = validate_extraction(raw)
    assert len(result.entities) == 2
    assert len(result.relationships) == 0
