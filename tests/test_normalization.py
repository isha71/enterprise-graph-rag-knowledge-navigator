from app.graph.normalization import normalize_entity_name, compute_entity_id
from app.models.domain import EntityType


def test_basic_normalization():
    result = normalize_entity_name("  Bob   Chen  ", EntityType.EMPLOYEE)
    assert result == "bob chen"


def test_vendor_suffix_removal():
    result = normalize_entity_name("GraphSphere Technologies Inc.", EntityType.VENDOR)
    assert result == "graphsphere technologies"


def test_vendor_multiple_suffixes():
    result = normalize_entity_name("Some Company LLC", EntityType.VENDOR)
    assert result == "some company"


def test_non_vendor_keeps_suffix():
    result = normalize_entity_name("Some Project Inc", EntityType.PROJECT)
    # Project type should NOT remove suffixes
    assert "inc" in result


def test_entity_id_determinism():
    id1 = compute_entity_id(EntityType.EMPLOYEE, "bob chen")
    id2 = compute_entity_id(EntityType.EMPLOYEE, "bob chen")
    assert id1 == id2
    assert len(id1) == 24


def test_entity_id_different_types():
    id1 = compute_entity_id(EntityType.EMPLOYEE, "bob")
    id2 = compute_entity_id(EntityType.PROJECT, "bob")
    assert id1 != id2


def test_unicode_normalization():
    result = normalize_entity_name("Café Company", EntityType.VENDOR)
    assert "café" in result or "cafe" in result
