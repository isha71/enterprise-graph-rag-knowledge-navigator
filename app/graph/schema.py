from app.models.domain import EntityType, RelationshipType

ALLOWED_ENTITY_TYPES = set(EntityType)
ALLOWED_RELATIONSHIP_TYPES = set(RelationshipType)

ALLOWED_RELATIONSHIP_SIGNATURES: dict[RelationshipType, set[tuple[EntityType, EntityType]]] = {
    RelationshipType.WORKS_IN: {
        (EntityType.EMPLOYEE, EntityType.TEAM),
    },
    RelationshipType.MANAGES: {
        (EntityType.EMPLOYEE, EntityType.TEAM),
    },
    RelationshipType.OWNS: {
        (EntityType.TEAM, EntityType.PROJECT),
    },
    RelationshipType.MAINTAINS: {
        (EntityType.EMPLOYEE, EntityType.PROJECT),
        (EntityType.TEAM, EntityType.SERVICE),
    },
    RelationshipType.USES: {
        (EntityType.PROJECT, EntityType.TECHNOLOGY),
        (EntityType.SERVICE, EntityType.TECHNOLOGY),
    },
    RelationshipType.DEPENDS_ON: {
        (EntityType.PROJECT, EntityType.SERVICE),
        (EntityType.SERVICE, EntityType.SERVICE),
    },
    RelationshipType.SUPPORTED_BY: {
        (EntityType.TECHNOLOGY, EntityType.VENDOR),
    },
    RelationshipType.AFFECTS: {
        (EntityType.INCIDENT, EntityType.SERVICE),
    },
    RelationshipType.GOVERNED_BY: {
        (EntityType.PROJECT, EntityType.POLICY),
    },
    RelationshipType.REPORTS_TO: {
        (EntityType.EMPLOYEE, EntityType.DEPARTMENT),
    },
}


def is_valid_entity_type(t: str) -> bool:
    try:
        EntityType(t)
        return True
    except ValueError:
        return False


def is_valid_relationship_type(t: str) -> bool:
    try:
        RelationshipType(t)
        return True
    except ValueError:
        return False


def is_valid_relationship_signature(
    relationship_type: RelationshipType,
    source_type: EntityType,
    target_type: EntityType,
) -> bool:
    """Check if a (source_type, target_type) pair is allowed for the relationship type."""
    return (source_type, target_type) in ALLOWED_RELATIONSHIP_SIGNATURES.get(
        relationship_type, set()
    )
