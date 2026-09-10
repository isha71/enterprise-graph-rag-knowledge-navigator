"""Tests for evaluation helper functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import (
    path_contains_ordered_entities,
    check_graph_path_hit,
    check_answer_accuracy,
    check_evidence_hit,
)


# --- path_contains_ordered_entities ---

def test_ordered_entities_correct_single_path():
    actual = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert path_contains_ordered_entities(actual, expected) is True


def test_ordered_entities_subsequence_with_extras():
    """Extra intermediate entities but correct ordered subsequence."""
    actual = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j", "GraphSphere Technologies"]
    expected = ["Bob Chen", "Project Atlas", "Neo4j"]
    assert path_contains_ordered_entities(actual, expected) is True


def test_ordered_entities_wrong_order():
    actual = ["Neo4j", "Project Atlas", "AI Platform Team", "Bob Chen"]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert path_contains_ordered_entities(actual, expected) is False


def test_ordered_entities_empty_expected():
    actual = ["Bob Chen", "AI Platform Team"]
    assert path_contains_ordered_entities(actual, []) is True


def test_ordered_entities_case_insensitive():
    actual = ["bob chen", "ai platform team"]
    expected = ["Bob Chen", "AI Platform Team"]
    assert path_contains_ordered_entities(actual, expected) is True


# --- check_graph_path_hit ---

def test_graph_path_hit_single_path_match():
    """One path contains all expected entities in order => True."""
    graph_evidence = [
        {
            "text": "Bob Chen -[MANAGES]-> AI Platform Team -[OWNS]-> Project Atlas",
            "metadata": {
                "path_nodes": [
                    {"name": "Bob Chen"},
                    {"name": "AI Platform Team"},
                    {"name": "Project Atlas"},
                ],
            },
        }
    ]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas"]
    assert check_graph_path_hit(graph_evidence, expected) is True


def test_graph_path_hit_entities_split_across_paths():
    """Expected entities distributed across different paths => False."""
    graph_evidence = [
        {
            "text": "Bob Chen -[MANAGES]-> AI Platform Team",
            "metadata": {
                "path_nodes": [
                    {"name": "Bob Chen"},
                    {"name": "AI Platform Team"},
                ],
            },
        },
        {
            "text": "Project Atlas -[USES]-> Neo4j",
            "metadata": {
                "path_nodes": [
                    {"name": "Project Atlas"},
                    {"name": "Neo4j"},
                ],
            },
        },
    ]
    expected = ["Bob Chen", "AI Platform Team", "Project Atlas", "Neo4j"]
    assert check_graph_path_hit(graph_evidence, expected) is False


def test_graph_path_hit_empty_expected():
    assert check_graph_path_hit([], []) is True


def test_graph_path_hit_no_evidence():
    assert check_graph_path_hit([], ["Bob Chen"]) is False


# --- check_answer_accuracy ---

def test_answer_accuracy_all_terms_present():
    answer = "Neo4j is used by Project Atlas."
    assert check_answer_accuracy(answer, ["Neo4j"]) is True


def test_answer_accuracy_missing_term():
    answer = "Project Atlas uses some database."
    assert check_answer_accuracy(answer, ["Neo4j"]) is False


def test_answer_accuracy_case_insensitive():
    answer = "the ai platform team owns it"
    assert check_answer_accuracy(answer, ["AI Platform Team"]) is True


# --- check_evidence_hit ---

def test_evidence_hit_present():
    evidence = [{"text": "Project Atlas uses Neo4j for graph storage."}]
    assert check_evidence_hit(evidence, ["Neo4j"]) is True


def test_evidence_hit_missing():
    evidence = [{"text": "Project Ledger uses PostgreSQL."}]
    assert check_evidence_hit(evidence, ["Neo4j"]) is False
