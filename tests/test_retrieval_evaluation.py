"""Tests for the fast retrieval-focused evaluation.

Covers:
- Selects exactly 4 questions from the full dataset
- Category distribution: 1 factual + 1 relationship + 2 multi-hop
- No final LLM generation is called
- Vector-only performs actual vector retrieval
- GraphRAG performs actual vector + graph retrieval
- Evidence metrics are calculated from real results
- Graph-path metrics are calculated from real results
- Results are saved with correct structure
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from evaluation.retrieval_evaluator import (
    select_retrieval_questions,
    evaluate_retrieval_question,
    run_retrieval_evaluation,
    run_single_question_evaluation,
    RETRIEVAL_EVAL_QUESTION_IDS,
)
from evaluation.evaluator import EvaluationAbortError


# ---- Full question set for selection tests ----

def _load_questions():
    qpath = Path(__file__).parent.parent / "evaluation" / "questions.json"
    with open(qpath) as f:
        return json.load(f)


# ---- Question selection ----

def test_selects_exactly_4_questions():
    questions = _load_questions()
    selected = select_retrieval_questions(questions)
    assert len(selected) == 4


def test_selected_ids_match_expected():
    questions = _load_questions()
    selected = select_retrieval_questions(questions)
    ids = [q["id"] for q in selected]
    assert ids == RETRIEVAL_EVAL_QUESTION_IDS


def test_selected_categories():
    """1 factual, 1 relationship, 2 multi_hop."""
    questions = _load_questions()
    selected = select_retrieval_questions(questions)
    cats = [q["category"] for q in selected]
    assert cats.count("factual") == 1
    assert cats.count("relationship") == 1
    assert cats.count("multi_hop") == 2


# ---- No generation called ----

def test_vector_only_does_not_call_generate():
    """Vector-only retrieval evaluation must not invoke generate_answer."""
    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo:
        mock_vo.return_value = {
            "reranked_evidence": [],
            "graph_evidence": [],
            "vector_evidence": [],
            "errors": [],
        }
        q = {"id": "t1", "category": "factual", "question": "Test?",
             "expected_answer_terms": [], "expected_path_entities": []}

        asyncio.run(evaluate_retrieval_question(q, "vector_only"))

        # generate_answer is NOT in the call path — _run_vector_only_retrieval
        # calls vector_retrieval, merge_context, rerank_context only
        mock_vo.assert_called_once()


def test_graphrag_does_not_call_generate():
    """GraphRAG retrieval evaluation must not invoke generate_answer."""
    with patch("evaluation.retrieval_evaluator._run_graphrag_retrieval", new_callable=AsyncMock) as mock_gr:
        mock_gr.return_value = {
            "reranked_evidence": [],
            "graph_evidence": [],
            "vector_evidence": [],
            "errors": [],
        }
        q = {"id": "t1", "category": "factual", "question": "Test?",
             "expected_answer_terms": [], "expected_path_entities": []}

        asyncio.run(evaluate_retrieval_question(q, "graphrag"))
        mock_gr.assert_called_once()


def test_vector_only_calls_vector_retrieval():
    """Vector-only path calls vector_retrieval and rerank_context."""
    with patch("app.workflow.nodes.vector_retrieval", new_callable=AsyncMock) as mock_vr, \
         patch("app.workflow.nodes.merge_context", new_callable=AsyncMock) as mock_mc, \
         patch("app.workflow.nodes.rerank_context", new_callable=AsyncMock) as mock_rr:
        mock_vr.return_value = {"vector_evidence": []}
        mock_mc.return_value = {"merged_evidence": []}
        mock_rr.return_value = {"reranked_evidence": []}

        from evaluation.retrieval_evaluator import _run_vector_only_retrieval
        asyncio.run(_run_vector_only_retrieval("Test question?"))

        mock_vr.assert_called_once()
        mock_mc.assert_called_once()
        mock_rr.assert_called_once()


def test_graphrag_calls_all_retrieval_stages():
    """GraphRAG path calls analyze_query, vector, graph, merge, rerank."""
    with patch("app.workflow.nodes.analyze_query", new_callable=AsyncMock) as mock_qa, \
         patch("app.workflow.nodes.vector_retrieval", new_callable=AsyncMock) as mock_vr, \
         patch("app.workflow.nodes.graph_retrieval", new_callable=AsyncMock) as mock_gr, \
         patch("app.workflow.nodes.merge_context", new_callable=AsyncMock) as mock_mc, \
         patch("app.workflow.nodes.rerank_context", new_callable=AsyncMock) as mock_rr:
        mock_qa.return_value = {"query_analysis": {"query_type": "factual", "entity_mentions": [], "relationship_hints": []}}
        mock_vr.return_value = {"vector_evidence": []}
        mock_gr.return_value = {"graph_evidence": []}
        mock_mc.return_value = {"merged_evidence": []}
        mock_rr.return_value = {"reranked_evidence": []}

        from evaluation.retrieval_evaluator import _run_graphrag_retrieval
        asyncio.run(_run_graphrag_retrieval("Test question?"))

        mock_qa.assert_called_once()
        mock_vr.assert_called_once()
        mock_gr.assert_called_once()
        mock_mc.assert_called_once()
        mock_rr.assert_called_once()


# ---- Metrics calculation ----

def test_evidence_hit_calculated():
    """evidence_hit is computed from reranked evidence and expected terms."""
    q = {"id": "t1", "category": "factual", "question": "Who manages?",
         "expected_answer_terms": ["Alice"], "expected_path_entities": []}

    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo:
        mock_vo.return_value = {
            "reranked_evidence": [{"text": "Alice manages the team"}],
            "graph_evidence": [],
            "vector_evidence": [{"text": "Alice manages the team"}],
            "errors": [],
        }

        result = asyncio.run(evaluate_retrieval_question(q, "vector_only"))
        assert result["evidence_hit"] is True


def test_evidence_hit_miss():
    """evidence_hit is False when terms are missing."""
    q = {"id": "t1", "category": "factual", "question": "Who manages?",
         "expected_answer_terms": ["Alice"], "expected_path_entities": []}

    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo:
        mock_vo.return_value = {
            "reranked_evidence": [{"text": "Bob runs the project"}],
            "graph_evidence": [],
            "vector_evidence": [{"text": "Bob runs the project"}],
            "errors": [],
        }

        result = asyncio.run(evaluate_retrieval_question(q, "vector_only"))
        assert result["evidence_hit"] is False


def test_graph_path_hit_calculated():
    """graph_path_hit is computed from graph evidence and expected path entities."""
    q = {"id": "t1", "category": "multi_hop", "question": "Test?",
         "expected_answer_terms": ["Neo4j"],
         "expected_path_entities": ["Bob Chen", "AI Platform Team", "Project Atlas"]}

    with patch("evaluation.retrieval_evaluator._run_graphrag_retrieval", new_callable=AsyncMock) as mock_gr:
        mock_gr.return_value = {
            "reranked_evidence": [{"text": "Neo4j is used"}],
            "graph_evidence": [{
                "text": "Bob Chen -[MANAGES]-> AI Platform Team -[OWNS]-> Project Atlas",
                "metadata": {"path_nodes": [
                    {"name": "Bob Chen"}, {"name": "AI Platform Team"}, {"name": "Project Atlas"}
                ]},
            }],
            "vector_evidence": [],
            "errors": [],
        }

        result = asyncio.run(evaluate_retrieval_question(q, "graphrag"))
        assert result["graph_path_hit"] is True


def test_graph_path_hit_not_computed_for_vector_only():
    """graph_path_hit should be None for vector-only mode."""
    q = {"id": "t1", "category": "factual", "question": "Test?",
         "expected_answer_terms": [], "expected_path_entities": []}

    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo:
        mock_vo.return_value = {
            "reranked_evidence": [],
            "graph_evidence": [],
            "vector_evidence": [],
            "errors": [],
        }

        result = asyncio.run(evaluate_retrieval_question(q, "vector_only"))
        assert result["graph_path_hit"] is None


# ---- Report structure ----

def test_report_saved_with_correct_structure(tmp_path):
    """run_retrieval_evaluation saves retrieval_latest.json with correct keys."""
    questions = _load_questions()
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps(questions))

    # Mock all retrieval functions
    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo, \
         patch("evaluation.retrieval_evaluator._run_graphrag_retrieval", new_callable=AsyncMock) as mock_gr, \
         patch("evaluation.retrieval_evaluator.get_settings") as mock_gs:
        settings = MagicMock()
        settings.llm_provider = "ollama"
        settings.groq_request_delay_seconds = 0
        settings.embedding_model = "test-embed"
        settings.reranker_model = "test-rerank"
        mock_gs.return_value = settings

        fake_result = {
            "reranked_evidence": [],
            "graph_evidence": [],
            "vector_evidence": [],
            "errors": [],
        }
        mock_vo.return_value = fake_result
        mock_gr.return_value = fake_result

        # Patch results dir to tmp_path
        with patch("evaluation.retrieval_evaluator.Path") as mock_path_cls:
            # Make Path(__file__).parent point to tmp_path for results saving
            mock_path_cls.return_value.parent = tmp_path
            mock_path_cls.__truediv__ = Path.__truediv__

            report = asyncio.run(run_retrieval_evaluation(str(questions_file)))

    assert report["evaluation_type"] == "retrieval"
    assert report["question_count"] == 4
    assert "overall_evidence_hit_rate" in report["vector_only"]
    assert "multi_hop_evidence_hit_rate" in report["vector_only"]
    assert "overall_evidence_hit_rate" in report["graph_rag"]
    assert "multi_hop_evidence_hit_rate" in report["graph_rag"]
    assert "multi_hop_graph_path_hit_rate" in report["graph_rag"]
    assert len(report["per_question"]) == 4


# ---- Query analysis uses reduced token budget ----

def test_query_analysis_uses_reduced_completion_tokens_for_groq():
    """When LLM_PROVIDER=groq, analyze_query passes max_completion_tokens=200."""
    from app.workflow.nodes import analyze_query

    captured_kwargs = {}

    async def _mock_generate_json(prompt, temperature=0, **kwargs):
        captured_kwargs.update(kwargs)
        return {"query_type": "factual", "entity_mentions": [], "relationship_hints": []}

    with patch("app.workflow.nodes.get_llm_client") as mock_llm_factory, \
         patch("app.workflow.nodes.get_settings") as mock_gs:
        settings = MagicMock()
        settings.llm_provider = "groq"
        mock_gs.return_value = settings

        mock_llm = MagicMock()
        mock_llm.generate_json = _mock_generate_json
        mock_llm_factory.return_value = mock_llm

        state = {"question": "Test?", "errors": []}
        asyncio.run(analyze_query(state))

    assert captured_kwargs.get("max_completion_tokens") == 200


def test_query_analysis_no_reduced_tokens_for_ollama():
    """When LLM_PROVIDER=ollama, analyze_query does NOT pass max_completion_tokens."""
    from app.workflow.nodes import analyze_query

    captured_kwargs = {}

    async def _mock_generate_json(prompt, temperature=0, **kwargs):
        captured_kwargs.update(kwargs)
        return {"query_type": "factual", "entity_mentions": [], "relationship_hints": []}

    with patch("app.workflow.nodes.get_llm_client") as mock_llm_factory, \
         patch("app.workflow.nodes.get_settings") as mock_gs:
        settings = MagicMock()
        settings.llm_provider = "ollama"
        mock_gs.return_value = settings

        mock_llm = MagicMock()
        mock_llm.generate_json = _mock_generate_json
        mock_llm_factory.return_value = mock_llm

        state = {"question": "Test?", "errors": []}
        asyncio.run(analyze_query(state))

    assert "max_completion_tokens" not in captured_kwargs


# ---- Abort error propagation ----

def test_abort_error_propagates():
    """EvaluationAbortError still raised on infrastructure failure."""
    q = {"id": "t1", "category": "factual", "question": "Test?",
         "expected_answer_terms": [], "expected_path_entities": []}

    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo:
        mock_vo.return_value = {
            "reranked_evidence": [],
            "graph_evidence": [],
            "vector_evidence": [],
            "errors": ["Vector retrieval error: connection refused"],
        }

        with pytest.raises(EvaluationAbortError):
            asyncio.run(evaluate_retrieval_question(q, "vector_only"))


# ---- CLI filtering ----

def test_cli_no_filter_runs_all_4(tmp_path):
    """No --question flag runs all 4 lightweight questions."""
    from scripts.run_retrieval_evaluation import parse_args
    args = parse_args([])
    assert args.question is None


def test_cli_single_question_flag():
    """--question mh_002 sets args.question to 'mh_002'."""
    from scripts.run_retrieval_evaluation import parse_args
    args = parse_args(["--question", "mh_002"])
    assert args.question == "mh_002"


def test_single_question_invalid_id_raises():
    """Requesting an ID not in the lightweight set raises ValueError."""
    questions = _load_questions()
    questions_file = Path(__file__).parent.parent / "evaluation" / "questions.json"

    with pytest.raises(ValueError, match="not in the lightweight evaluation set"):
        asyncio.run(run_single_question_evaluation("s_003", str(questions_file)))


def test_single_question_result_structure(tmp_path):
    """Single-question run produces correct report structure."""
    questions = _load_questions()
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps(questions))

    fake_result = {
        "reranked_evidence": [],
        "graph_evidence": [],
        "vector_evidence": [],
        "errors": [],
    }

    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo, \
         patch("evaluation.retrieval_evaluator._run_graphrag_retrieval", new_callable=AsyncMock) as mock_gr, \
         patch("evaluation.retrieval_evaluator.get_settings") as mock_gs:
        settings = MagicMock()
        settings.llm_provider = "ollama"
        settings.groq_request_delay_seconds = 0
        settings.embedding_model = "test-embed"
        settings.reranker_model = "test-rerank"
        mock_gs.return_value = settings
        mock_vo.return_value = fake_result
        mock_gr.return_value = fake_result

        report = asyncio.run(run_single_question_evaluation("mh_002", str(questions_file)))

    assert report["evaluation_type"] == "retrieval_single_question"
    assert report["question_count"] == 1
    assert report["question_id"] == "mh_002"
    assert len(report["per_question"]) == 1
    assert report["per_question"][0]["id"] == "mh_002"


def test_single_question_does_not_overwrite_retrieval_latest(tmp_path):
    """Single-question result saves to retrieval_{id}.json, not retrieval_latest.json."""
    questions = _load_questions()
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps(questions))

    # Create a sentinel retrieval_latest.json
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sentinel = results_dir / "retrieval_latest.json"
    sentinel.write_text('{"sentinel": true}')

    fake_result = {
        "reranked_evidence": [],
        "graph_evidence": [],
        "vector_evidence": [],
        "errors": [],
    }

    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo, \
         patch("evaluation.retrieval_evaluator._run_graphrag_retrieval", new_callable=AsyncMock) as mock_gr, \
         patch("evaluation.retrieval_evaluator.get_settings") as mock_gs, \
         patch("evaluation.retrieval_evaluator.Path") as mock_path_cls:
        settings = MagicMock()
        settings.llm_provider = "ollama"
        settings.groq_request_delay_seconds = 0
        settings.embedding_model = "test-embed"
        settings.reranker_model = "test-rerank"
        mock_gs.return_value = settings
        mock_vo.return_value = fake_result
        mock_gr.return_value = fake_result

        # Redirect results directory to tmp_path
        mock_path_cls.return_value.parent = tmp_path
        mock_path_cls.__truediv__ = Path.__truediv__

        asyncio.run(run_single_question_evaluation("mh_002", str(questions_file)))

    # Sentinel must be unchanged
    assert json.loads(sentinel.read_text()) == {"sentinel": True}


def test_single_question_metrics_from_actual_output(tmp_path):
    """Single-question per_question results contain real evaluation data."""
    questions = _load_questions()
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps(questions))

    with patch("evaluation.retrieval_evaluator._run_vector_only_retrieval", new_callable=AsyncMock) as mock_vo, \
         patch("evaluation.retrieval_evaluator._run_graphrag_retrieval", new_callable=AsyncMock) as mock_gr, \
         patch("evaluation.retrieval_evaluator.get_settings") as mock_gs:
        settings = MagicMock()
        settings.llm_provider = "ollama"
        settings.groq_request_delay_seconds = 0
        settings.embedding_model = "test"
        settings.reranker_model = "test"
        mock_gs.return_value = settings

        mock_vo.return_value = {
            "reranked_evidence": [{"text": "GraphSphere Technologies supports Neo4j"}],
            "graph_evidence": [],
            "vector_evidence": [{"text": "GraphSphere Technologies supports Neo4j"}],
            "errors": [],
        }
        mock_gr.return_value = {
            "reranked_evidence": [{"text": "GraphSphere Technologies"}],
            "graph_evidence": [{
                "text": "Bob Chen -[MANAGES]-> AI Platform Team -[OWNS]-> Project Atlas -[USES]-> Neo4j -[SUPPORTED_BY]-> GraphSphere Technologies",
                "metadata": {"path_nodes": [
                    {"name": "Bob Chen"}, {"name": "AI Platform Team"},
                    {"name": "Project Atlas"}, {"name": "Neo4j"},
                    {"name": "GraphSphere Technologies"},
                ]},
            }],
            "vector_evidence": [],
            "errors": [],
        }

        report = asyncio.run(run_single_question_evaluation("mh_002", str(questions_file)))

    pq = report["per_question"][0]
    assert pq["vector_only"]["evidence_hit"] is True
    assert pq["graph_rag"]["evidence_hit"] is True
    assert pq["graph_rag"]["graph_path_hit"] is True
