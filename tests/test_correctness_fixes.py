"""Tests for the pre-evaluation correctness and integrity fixes.

Covers:
- Graph path hit rate denominator (excludes empty expected_path_entities)
- Graph extraction failure raises ExtractionError
- Neo4j persist_extraction propagates infrastructure errors
- Evaluation infrastructure error detection
- Question r_004 expected_path_entities fix
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.exceptions import ExtractionError, GraphQueryError
from app.models.domain import (
    EntityType, RelationshipType, ExtractedEntity,
    ExtractedRelationship, GraphExtraction,
)


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


# ---- Graph path hit rate denominator ----

def _make_question_result(qid, category, expected_path, graph_path_hit, answer_correct=True):
    return {
        "id": qid,
        "category": category,
        "expected_path_entities": expected_path,
        "vector_only": {"answer_correct": answer_correct, "evidence_hit": True},
        "graph_rag": {
            "answer_correct": answer_correct,
            "evidence_hit": True,
            "graph_path_hit": graph_path_hit,
        },
    }


def test_calc_path_rate_excludes_empty_expected():
    """calc_path_rate should only count questions with non-empty expected_path_entities."""
    from evaluation.evaluator import run_evaluation
    import evaluation.evaluator as ev_mod

    results = [
        _make_question_result("f_001", "factual", [], True),      # empty path => excluded
        _make_question_result("f_002", "factual", [], True),      # empty path => excluded
        _make_question_result("mh_001", "multi_hop", ["A", "B"], True),   # counted, hit
        _make_question_result("mh_002", "multi_hop", ["X", "Y"], False),  # counted, miss
    ]

    rate = ev_mod.calc_path_rate(results)
    assert rate == 0.5, f"Expected 0.5, got {rate}"


def test_calc_path_rate_all_empty_returns_zero():
    """When all questions have empty expected_path_entities, rate is 0.0."""
    import evaluation.evaluator as ev_mod

    results = [
        _make_question_result("f_001", "factual", [], True),
        _make_question_result("f_002", "factual", [], True),
    ]
    rate = ev_mod.calc_path_rate(results)
    assert rate == 0.0


def test_calc_path_rate_category_filter():
    """calc_path_rate with category filter only includes matching category."""
    import evaluation.evaluator as ev_mod

    results = [
        _make_question_result("r_001", "relationship", ["A", "B"], True),
        _make_question_result("mh_001", "multi_hop", ["C", "D"], False),
    ]
    rate = ev_mod.calc_path_rate(results, "relationship")
    assert rate == 1.0

    rate = ev_mod.calc_path_rate(results, "multi_hop")
    assert rate == 0.0


# ---- Graph extraction failure raises ExtractionError ----

def test_extract_graph_raises_on_failure():
    """extract_graph_from_text must raise ExtractionError when LLM fails twice."""
    from app.graph.extraction import extract_graph_from_text

    mock_llm = AsyncMock()
    mock_llm.generate_json.side_effect = RuntimeError("LLM down")
    mock_llm.generate.side_effect = RuntimeError("LLM down")

    with pytest.raises(ExtractionError, match="Graph extraction failed after retry"):
        _run(extract_graph_from_text("some text", mock_llm))


def test_extract_graph_success_returns_extraction():
    """extract_graph_from_text should return GraphExtraction on success."""
    from app.graph.extraction import extract_graph_from_text

    mock_llm = AsyncMock()
    mock_llm.generate_json.return_value = {
        "entities": [{"name": "ProjectX", "type": "Project"}],
        "relationships": [],
    }

    result = _run(extract_graph_from_text("some text", mock_llm))
    assert isinstance(result, GraphExtraction)
    assert len(result.entities) == 1
    assert result.entities[0].name == "ProjectX"


def test_extract_graph_retry_success():
    """First attempt fails, retry succeeds."""
    from app.graph.extraction import extract_graph_from_text

    mock_llm = AsyncMock()
    mock_llm.generate_json.side_effect = [
        RuntimeError("bad json"),
        {"entities": [{"name": "A", "type": "Project"}], "relationships": []},
    ]
    mock_llm.generate.return_value = '{"entities": [{"name": "A", "type": "Project"}], "relationships": []}'

    result = _run(extract_graph_from_text("text", mock_llm))
    assert len(result.entities) == 1


# ---- Neo4j persist_extraction: propagate infrastructure errors ----

def test_persist_extraction_propagates_connection_error():
    """Infrastructure errors (not GraphQueryError) must propagate."""
    from app.graph.neo4j_store import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)

    entity = ExtractedEntity(name="TestEntity", type=EntityType.PROJECT)
    extraction = GraphExtraction(entities=[entity], relationships=[])

    async def _test():
        with patch.object(store, "upsert_entity", side_effect=ConnectionError("Neo4j down")):
            with pytest.raises(ConnectionError):
                await store.persist_extraction(extraction, "doc1", "chunk1", "test.md")

    _run(_test())


def test_persist_extraction_skips_validation_error():
    """GraphQueryError (validation) should be logged and skipped, not propagated."""
    from app.graph.neo4j_store import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)

    entity = ExtractedEntity(name="TestEntity", type=EntityType.PROJECT)
    extraction = GraphExtraction(entities=[entity], relationships=[])

    async def _test():
        with patch.object(store, "upsert_entity", side_effect=GraphQueryError("invalid type")):
            e_count, r_count = await store.persist_extraction(extraction, "doc1", "chunk1", "test.md")
        assert e_count == 0
        assert r_count == 0

    _run(_test())


def test_persist_extraction_rel_propagates_connection_error():
    """Infrastructure errors on relationship upsert must also propagate."""
    from app.graph.neo4j_store import Neo4jStore

    store = Neo4jStore.__new__(Neo4jStore)

    e1 = ExtractedEntity(name="TeamA", type=EntityType.TEAM)
    e2 = ExtractedEntity(name="ProjectX", type=EntityType.PROJECT)
    rel = ExtractedRelationship(source="TeamA", type=RelationshipType.OWNS, target="ProjectX")
    extraction = GraphExtraction(entities=[e1, e2], relationships=[rel])

    async def _test():
        with (
            patch.object(store, "upsert_entity", new_callable=AsyncMock),
            patch.object(store, "upsert_relationship", side_effect=ConnectionError("Neo4j down")),
        ):
            with pytest.raises(ConnectionError):
                await store.persist_extraction(extraction, "doc1", "chunk1", "test.md")

    _run(_test())


# ---- Evaluation abort on infrastructure errors ----

def test_has_critical_errors_detects_vector_failure():
    """Critical error keywords should be detected."""
    from evaluation.evaluator import _has_critical_errors

    errors = ["Vector retrieval error: connection refused"]
    critical = _has_critical_errors(errors, "vector_only")
    assert len(critical) == 1


def test_has_critical_errors_ignores_graph_in_vector_mode():
    """Graph retrieval error is not critical for vector_only mode."""
    from evaluation.evaluator import _has_critical_errors

    errors = ["Graph retrieval error: Neo4j down"]
    critical = _has_critical_errors(errors, "vector_only")
    assert len(critical) == 0


def test_has_critical_errors_detects_graph_in_graphrag_mode():
    """Graph retrieval error IS critical for graphrag mode."""
    from evaluation.evaluator import _has_critical_errors

    errors = ["Graph retrieval error: Neo4j down"]
    critical = _has_critical_errors(errors, "graphrag")
    assert len(critical) == 1


def test_has_critical_errors_no_errors():
    from evaluation.evaluator import _has_critical_errors
    assert _has_critical_errors([], "graphrag") == []


# ---- Question r_004 expected_path_entities fix ----

def test_r004_has_empty_expected_path_entities():
    """r_004 asks about two separate governance relationships, not a single path."""
    questions_path = Path(__file__).parent.parent / "evaluation" / "questions.json"
    with open(questions_path) as f:
        questions = json.load(f)

    r004 = next(q for q in questions if q["id"] == "r_004")
    assert r004["expected_path_entities"] == [], (
        f"r_004 should have empty expected_path_entities, got {r004['expected_path_entities']}"
    )


# ---- DocumentUploadResponse has graph chunk fields ----

def test_document_upload_response_has_graph_chunk_fields():
    from app.models.schemas import DocumentUploadResponse

    resp = DocumentUploadResponse(
        document_id="abc",
        document_name="test.md",
        chunks_created=5,
        vector_points_upserted=5,
        entities_extracted=10,
        relationships_extracted=3,
        graph_chunks_attempted=5,
        graph_chunks_succeeded=4,
        graph_chunks_failed=1,
    )
    assert resp.graph_chunks_attempted == 5
    assert resp.graph_chunks_succeeded == 4
    assert resp.graph_chunks_failed == 1


def test_document_upload_response_defaults():
    """Graph chunk fields default to 0 for backward compat."""
    from app.models.schemas import DocumentUploadResponse

    resp = DocumentUploadResponse(
        document_id="abc",
        document_name="test.md",
        chunks_created=5,
        vector_points_upserted=5,
        entities_extracted=10,
        relationships_extracted=3,
    )
    assert resp.graph_chunks_attempted == 0
    assert resp.graph_chunks_succeeded == 0
    assert resp.graph_chunks_failed == 0
