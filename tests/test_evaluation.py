"""Tests for evaluation utility functions."""
from evaluation.evaluator import (
    path_contains_ordered_entities,
    check_graph_path_hit,
    check_answer_accuracy,
    check_evidence_hit,
)


def test_ordered_entities_exact_match():
    actual = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert path_contains_ordered_entities(actual, expected) is True


def test_ordered_entities_subsequence():
    actual = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    expected = ["Bob Chen", "Project Atlas", "Neo4j"]
    assert path_contains_ordered_entities(actual, expected) is True


def test_ordered_entities_wrong_order():
    actual = ["Neo4j", "Project Atlas", "AI Platform Team", "Bob Chen"]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert path_contains_ordered_entities(actual, expected) is False


def test_ordered_entities_case_insensitive():
    actual = ["bob chen", "AI Platform Team"]
    expected = ["Bob Chen", "ai platform team"]
    assert path_contains_ordered_entities(actual, expected) is True


def test_ordered_entities_empty_expected():
    assert path_contains_ordered_entities(["A", "B"], []) is True


def test_ordered_entities_missing():
    actual = ["Bob Chen", "AI Platform Team"]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert path_contains_ordered_entities(actual, expected) is False


def test_graph_path_hit_single_path_match():
    """One path that contains all expected entities in order passes."""
    evidence = [
        {
            "text": "Bob Chen -[MANAGES]-> AI Platform Team -[OWNS]-> Project Atlas -[USES]-> Neo4j",
            "metadata": {
                "path_nodes": [
                    {"name": "Bob Chen", "type": "Employee"},
                    {"name": "AI Platform Team", "type": "Team"},
                    {"name": "Project Atlas", "type": "Project"},
                    {"name": "Neo4j", "type": "Technology"},
                ]
            }
        }
    ]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert check_graph_path_hit(evidence, expected) is True


def test_graph_path_hit_scattered_fails():
    """Two separate paths that individually cover different parts must NOT pass."""
    evidence = [
        {
            "text": "Bob Chen -[MANAGES]-> AI Platform Team",
            "metadata": {
                "path_nodes": [
                    {"name": "Bob Chen", "type": "Employee"},
                    {"name": "AI Platform Team", "type": "Team"},
                ]
            }
        },
        {
            "text": "Project Atlas -[USES]-> Neo4j",
            "metadata": {
                "path_nodes": [
                    {"name": "Project Atlas", "type": "Project"},
                    {"name": "Neo4j", "type": "Technology"},
                ]
            }
        },
    ]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert check_graph_path_hit(evidence, expected) is False


def test_graph_path_hit_empty_expected():
    assert check_graph_path_hit([], []) is True


def test_answer_accuracy_basic():
    assert check_answer_accuracy("The AI Platform Team owns Project Atlas", ["AI Platform Team"]) is True


def test_answer_accuracy_missing():
    assert check_answer_accuracy("Project Atlas", ["Neo4j"]) is False


def test_evidence_hit():
    evidence = [{"text": "Neo4j is a graph database"}]
    assert check_evidence_hit(evidence, ["Neo4j"]) is True
    assert check_evidence_hit(evidence, ["PostgreSQL"]) is False
