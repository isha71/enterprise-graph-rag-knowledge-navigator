"""Tests for query type normalization in workflow nodes."""
from app.workflow.nodes import _normalize_query_type
from app.models.domain import QueryType


def test_normalize_standard_values():
    assert _normalize_query_type("factual") == QueryType.FACTUAL
    assert _normalize_query_type("semantic") == QueryType.SEMANTIC
    assert _normalize_query_type("relationship") == QueryType.RELATIONSHIP
    assert _normalize_query_type("multi_hop") == QueryType.MULTI_HOP


def test_normalize_hyphenated():
    assert _normalize_query_type("multi-hop") == QueryType.MULTI_HOP


def test_normalize_uppercase():
    assert _normalize_query_type("MULTI_HOP") == QueryType.MULTI_HOP
    assert _normalize_query_type("FACTUAL") == QueryType.FACTUAL


def test_normalize_mixed_case_with_spaces():
    assert _normalize_query_type("Multi Hop") == QueryType.MULTI_HOP
    assert _normalize_query_type("multi hop") == QueryType.MULTI_HOP


def test_normalize_with_whitespace():
    assert _normalize_query_type("  factual  ") == QueryType.FACTUAL


def test_normalize_invalid_falls_back_to_factual():
    assert _normalize_query_type("unknown") == QueryType.FACTUAL
    assert _normalize_query_type("complex") == QueryType.FACTUAL
    assert _normalize_query_type("") == QueryType.FACTUAL
