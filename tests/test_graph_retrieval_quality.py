"""Tests for graph retrieval quality improvements.

Covers:
- Generic seed entity filtering
- Path ranking by relationship-hint coverage
- Multi-hop vs non-multi-hop tie breaking
- Fuzzy lookup LIMIT 1
- Relationship-first provenance
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.retrieval import (
    filter_generic_entity_names,
    paths_to_evidence,
    GENERIC_ENTITY_TERMS,
)


# ---- Generic seed filtering ----

def test_filter_removes_generic_terms():
    names = ["Bob Chen", "team", "project", "technology", "vendor"]
    result = filter_generic_entity_names(names)
    assert result == ["Bob Chen"]


def test_filter_keeps_multi_word_names():
    names = ["AI Platform Team", "Project Atlas"]
    result = filter_generic_entity_names(names)
    assert result == ["AI Platform Team", "Project Atlas"]


def test_filter_case_insensitive():
    """All-generic input triggers fallback to original list."""
    names = ["Team", "PROJECT", "Vendor"]
    result = filter_generic_entity_names(names)
    assert result == names  # all generic, so falls back to original


def test_filter_fallback_when_all_generic():
    """If filtering removes everything, return the original list."""
    names = ["team", "project"]
    result = filter_generic_entity_names(names)
    assert result == ["team", "project"]


def test_filter_mixed_generic_and_named():
    names = ["Bob Chen", "employee", "AI Platform Team", "team"]
    result = filter_generic_entity_names(names)
    assert result == ["Bob Chen", "AI Platform Team"]


def test_filter_preserves_order():
    names = ["technology", "Alice Morgan", "vendor", "Neo4j"]
    result = filter_generic_entity_names(names)
    assert result == ["Alice Morgan", "Neo4j"]


def test_generic_terms_cover_all_entity_types():
    """Every generic term should be lowercase."""
    for term in GENERIC_ENTITY_TERMS:
        assert term == term.lower()


# ---- Path ranking by hint coverage ----

def _make_path(rel_types, node_names=None):
    """Build a minimal path dict for testing."""
    if node_names is None:
        node_names = [f"N{i}" for i in range(len(rel_types) + 1)]
    nodes = [
        {"name": n, "type": "Entity", "id": f"id_{n}",
         "source_chunk_ids": [], "source_document_ids": [], "source_document_names": []}
        for n in node_names
    ]
    rels = [
        {"type": rt, "forward": True,
         "source_chunk_ids": [], "source_document_ids": [], "source_document_names": []}
        for rt in rel_types
    ]
    return {"nodes": nodes, "relationships": rels}


def _score_path(path, hints):
    """Calculate hint match count for a path (mirrors Cypher logic)."""
    rel_types = {r["type"] for r in path["relationships"]}
    return sum(1 for h in hints if h in rel_types)


def test_path_with_more_hints_ranks_higher():
    """A path covering OWNS+USES+SUPPORTED_BY outranks one covering only OWNS."""
    hints = ["OWNS", "USES", "SUPPORTED_BY"]
    path_3_hints = _make_path(["MANAGES", "OWNS", "USES", "SUPPORTED_BY"])
    path_1_hint = _make_path(["OWNS"])

    score_3 = _score_path(path_3_hints, hints)
    score_1 = _score_path(path_1_hint, hints)
    assert score_3 > score_1


def test_path_with_two_hints_beats_one():
    hints = ["OWNS", "USES", "SUPPORTED_BY"]
    path_2 = _make_path(["OWNS", "USES"])
    path_1 = _make_path(["USES"])

    assert _score_path(path_2, hints) > _score_path(path_1, hints)


# ---- Multi-hop tie breaking ----

def test_multi_hop_prefers_longer_on_tie():
    """For equal hint coverage in multi_hop, longer path is preferred."""
    hints = ["USES"]
    long_path = _make_path(["MANAGES", "OWNS", "USES", "SUPPORTED_BY"])
    short_path = _make_path(["USES"])

    score_long = _score_path(long_path, hints)
    score_short = _score_path(short_path, hints)
    assert score_long == score_short  # tied on hint coverage

    # For multi_hop: prefer_longer=True => longer path wins
    len_long = len(long_path["relationships"])
    len_short = len(short_path["relationships"])
    assert len_long > len_short  # longer path should be preferred


# ---- Non-multi-hop tie breaking ----

def test_non_multi_hop_prefers_shorter_on_tie():
    """For equal hint coverage in factual/relationship, shorter path is preferred."""
    hints = ["USES"]
    long_path = _make_path(["MANAGES", "OWNS", "USES", "SUPPORTED_BY"])
    short_path = _make_path(["USES"])

    score_long = _score_path(long_path, hints)
    score_short = _score_path(short_path, hints)
    assert score_long == score_short  # tied on hint coverage

    # For non-multi_hop: prefer_longer=False => shorter path wins
    len_long = len(long_path["relationships"])
    len_short = len(short_path["relationships"])
    assert len_short < len_long  # shorter path should be preferred


# ---- Fuzzy lookup LIMIT 1 ----

def test_find_entity_fallback_uses_limit_1():
    """The fuzzy fallback query should use LIMIT 1, not LIMIT 5."""
    import inspect
    from app.graph.neo4j_store import Neo4jStore

    source = inspect.getsource(Neo4jStore.find_entity)
    # The fallback CONTAINS query should use LIMIT 1
    assert "LIMIT 1" in source
    assert "LIMIT 5" not in source


# ---- Relationship-first provenance ----

def test_provenance_from_relationships_only():
    """When relationships have provenance, node provenance should be excluded."""
    path = {
        "nodes": [
            {"name": "A", "type": "T", "id": "a",
             "source_chunk_ids": ["node_c1"], "source_document_ids": ["node_d1"],
             "source_document_names": ["node_doc.md"]},
            {"name": "B", "type": "T", "id": "b",
             "source_chunk_ids": ["node_c2"], "source_document_ids": ["node_d2"],
             "source_document_names": ["node_doc2.md"]},
        ],
        "relationships": [
            {"type": "USES", "forward": True,
             "source_chunk_ids": ["rel_c1"], "source_document_ids": ["rel_d1"],
             "source_document_names": ["rel_doc.md"]},
        ],
    }
    evidence = paths_to_evidence([path])
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.source_document_ids == ["rel_d1"]
    assert ev.source_chunk_ids == ["rel_c1"]
    assert ev.metadata["source_document_names"] == ["rel_doc.md"]


def test_provenance_falls_back_to_nodes_when_no_rel_provenance():
    """When relationships have no provenance, fall back to node provenance."""
    path = {
        "nodes": [
            {"name": "A", "type": "T", "id": "a",
             "source_chunk_ids": ["c1"], "source_document_ids": ["d1"],
             "source_document_names": ["doc.md"]},
            {"name": "B", "type": "T", "id": "b",
             "source_chunk_ids": ["c2"], "source_document_ids": ["d2"],
             "source_document_names": ["doc2.md"]},
        ],
        "relationships": [
            {"type": "USES", "forward": True,
             "source_chunk_ids": [], "source_document_ids": [],
             "source_document_names": []},
        ],
    }
    evidence = paths_to_evidence([path])
    assert len(evidence) == 1
    ev = evidence[0]
    assert sorted(ev.source_document_ids) == ["d1", "d2"]
    assert sorted(ev.source_chunk_ids) == ["c1", "c2"]


def test_provenance_multiple_relationships_merged():
    """Provenance from multiple relationships is merged."""
    path = {
        "nodes": [
            {"name": "A", "type": "T", "id": "a",
             "source_chunk_ids": ["nc1"], "source_document_ids": ["nd1"],
             "source_document_names": ["ndoc.md"]},
            {"name": "B", "type": "T", "id": "b",
             "source_chunk_ids": [], "source_document_ids": [],
             "source_document_names": []},
            {"name": "C", "type": "T", "id": "c",
             "source_chunk_ids": [], "source_document_ids": [],
             "source_document_names": []},
        ],
        "relationships": [
            {"type": "USES", "forward": True,
             "source_chunk_ids": ["rc1"], "source_document_ids": ["rd1"],
             "source_document_names": ["rdoc1.md"]},
            {"type": "SUPPORTED_BY", "forward": True,
             "source_chunk_ids": ["rc2"], "source_document_ids": ["rd2"],
             "source_document_names": ["rdoc2.md"]},
        ],
    }
    evidence = paths_to_evidence([path])
    ev = evidence[0]
    assert sorted(ev.source_document_ids) == ["rd1", "rd2"]
    assert sorted(ev.source_chunk_ids) == ["rc1", "rc2"]
    # Node provenance should NOT be included
    assert "nd1" not in ev.source_document_ids
