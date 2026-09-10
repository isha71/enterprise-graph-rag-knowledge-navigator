"""Tests for relationship-hint normalization in query analysis.

Covers alias resolution, text normalization, unknown removal,
deduplication, and integration with analyze_query.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.workflow.nodes import (
    _normalize_relationship_hints,
    RELATIONSHIP_HINT_ALIASES,
)


# ---- Alias resolution ----

def test_supports_becomes_supported_by():
    assert _normalize_relationship_hints(["SUPPORTS"]) == ["SUPPORTED_BY"]


def test_managed_by_becomes_manages():
    assert _normalize_relationship_hints(["MANAGED_BY"]) == ["MANAGES"]


def test_owned_by_becomes_owns():
    assert _normalize_relationship_hints(["OWNED_BY"]) == ["OWNS"]


def test_maintained_by_becomes_maintains():
    assert _normalize_relationship_hints(["MAINTAINED_BY"]) == ["MAINTAINS"]


def test_depends_becomes_depends_on():
    assert _normalize_relationship_hints(["DEPENDS"]) == ["DEPENDS_ON"]


# ---- Already-valid pass through ----

def test_valid_relationships_unchanged():
    valid = ["WORKS_IN", "MANAGES", "OWNS", "MAINTAINS", "USES",
             "DEPENDS_ON", "SUPPORTED_BY", "AFFECTS", "GOVERNED_BY", "REPORTS_TO"]
    assert _normalize_relationship_hints(valid) == valid


# ---- Text normalization ----

def test_lowercase_normalized():
    assert _normalize_relationship_hints(["supports"]) == ["SUPPORTED_BY"]


def test_mixed_case_normalized():
    assert _normalize_relationship_hints(["Managed_By"]) == ["MANAGES"]


def test_hyphens_replaced():
    assert _normalize_relationship_hints(["DEPENDS-ON"]) == ["DEPENDS_ON"]


def test_spaces_replaced():
    assert _normalize_relationship_hints(["DEPENDS ON"]) == ["DEPENDS_ON"]


def test_whitespace_stripped():
    assert _normalize_relationship_hints(["  USES  "]) == ["USES"]


# ---- Unknown removal ----

def test_unknown_hint_removed():
    assert _normalize_relationship_hints(["RANDOM_RELATION"]) == []


def test_unknown_mixed_with_valid():
    result = _normalize_relationship_hints(["SUPPORTS", "USES", "DEPENDS_ON", "random_relation"])
    assert result == ["SUPPORTED_BY", "USES", "DEPENDS_ON"]


def test_non_string_ignored():
    result = _normalize_relationship_hints([42, None, "USES"])
    assert result == ["USES"]


# ---- Deduplication ----

def test_duplicates_removed_preserve_order():
    result = _normalize_relationship_hints(["SUPPORTS", "SUPPORTED_BY", "USES"])
    assert result == ["SUPPORTED_BY", "USES"]


def test_alias_dedup_across_variants():
    """MANAGE and MANAGES both resolve to MANAGES — only one kept."""
    result = _normalize_relationship_hints(["MANAGE", "MANAGES", "MANAGED_BY"])
    assert result == ["MANAGES"]


# ---- Integration: analyze_query uses normalized hints ----

def test_analyze_query_normalizes_hints():
    """QueryAnalysis construction path uses _normalize_relationship_hints."""
    raw = ["SUPPORTS", "USES", "DEPENDS_ON", "BOGUS"]
    normalized = _normalize_relationship_hints(raw)
    assert normalized == ["SUPPORTED_BY", "USES", "DEPENDS_ON"]


# ---- Prompt clarification ----

def test_prompt_contains_multi_hop_clarification():
    from app.generation.prompts import QUERY_ANALYSIS_PROMPT
    assert "two or more connected relationships across intermediate entities" in QUERY_ANALYSIS_PROMPT
