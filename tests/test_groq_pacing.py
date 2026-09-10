"""Tests for Groq free-tier pacing in the evaluation runner.

Covers:
- Groq evaluation uses the configured delay
- Delay occurs between vector_only and graphrag runs
- Delay occurs before moving to the next question
- Ollama/OpenAI evaluation does not use Groq pacing
- EvaluationAbortError behaviour remains intact
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from evaluation.evaluator import run_evaluation, EvaluationAbortError


def _make_question(qid: str, category: str = "factual"):
    return {
        "id": qid,
        "category": category,
        "question": f"Test question {qid}?",
        "expected_answer_terms": ["answer"],
        "expected_path_entities": [],
    }


def _fake_query_result():
    return {
        "answer": "answer text",
        "reranked_evidence": [],
        "graph_evidence": [],
        "vector_evidence": [],
        "errors": [],
    }


TWO_QUESTIONS = [_make_question("q1"), _make_question("q2")]


@pytest.fixture
def _mock_env(tmp_path):
    """Patch settings, run_query, and questions file for evaluation tests."""
    questions_file = tmp_path / "questions.json"
    import json
    questions_file.write_text(json.dumps(TWO_QUESTIONS))

    with patch("evaluation.evaluator.run_query", new_callable=AsyncMock) as mock_rq, \
         patch("evaluation.evaluator.get_settings") as mock_gs:
        settings = MagicMock()
        settings.llm_provider = "groq"
        settings.groq_request_delay_seconds = 0.01
        settings.embedding_model = "test"
        settings.reranker_model = "test"
        settings.ollama_model = "test"
        settings.openai_model = ""
        mock_gs.return_value = settings
        mock_rq.return_value = _fake_query_result()
        yield settings, mock_rq, str(questions_file)


def test_groq_pacing_sleeps_between_modes(_mock_env):
    """Groq evaluation sleeps between vector_only and graphrag runs."""
    settings, mock_rq, qpath = _mock_env
    sleep_calls = []
    original_sleep = asyncio.sleep

    async def _tracking_sleep(secs):
        sleep_calls.append(secs)

    with patch("evaluation.evaluator.asyncio.sleep", side_effect=_tracking_sleep):
        asyncio.run(run_evaluation(qpath))

    # 2 questions: sleep after q1 vector_only (before q1 graphrag),
    # sleep before q2 (between questions), sleep after q2 vector_only
    assert len(sleep_calls) == 3
    assert all(s == settings.groq_request_delay_seconds for s in sleep_calls)


def test_groq_pacing_uses_configured_delay(_mock_env):
    """Delay value comes from settings.groq_request_delay_seconds."""
    settings, mock_rq, qpath = _mock_env
    settings.groq_request_delay_seconds = 0.02
    sleep_calls = []

    async def _tracking_sleep(secs):
        sleep_calls.append(secs)

    with patch("evaluation.evaluator.asyncio.sleep", side_effect=_tracking_sleep):
        asyncio.run(run_evaluation(qpath))

    assert all(s == 0.02 for s in sleep_calls)


def test_groq_pacing_delay_between_vector_and_graphrag(_mock_env):
    """For each question, sleep occurs after vector_only before graphrag."""
    settings, mock_rq, qpath = _mock_env
    events = []

    async def _tracking_rq(question, mode="graphrag"):
        events.append(("query", mode))
        return _fake_query_result()

    async def _tracking_sleep(secs):
        events.append(("sleep", secs))

    mock_rq.side_effect = _tracking_rq
    with patch("evaluation.evaluator.asyncio.sleep", side_effect=_tracking_sleep):
        asyncio.run(run_evaluation(qpath))

    # q1: query vector_only -> sleep -> query graphrag -> sleep (before q2) -> q2: query vector_only -> sleep -> query graphrag
    assert events[0] == ("query", "vector_only")
    assert events[1][0] == "sleep"
    assert events[2] == ("query", "graphrag")


def test_groq_pacing_delay_before_next_question(_mock_env):
    """Sleep occurs before starting the next question."""
    settings, mock_rq, qpath = _mock_env
    events = []

    async def _tracking_rq(question, mode="graphrag"):
        events.append(("query", mode))
        return _fake_query_result()

    async def _tracking_sleep(secs):
        events.append(("sleep", secs))

    mock_rq.side_effect = _tracking_rq
    with patch("evaluation.evaluator.asyncio.sleep", side_effect=_tracking_sleep):
        asyncio.run(run_evaluation(qpath))

    # After q1 graphrag, there should be a sleep before q2 vector_only
    # events: q1_vo, sleep, q1_gr, sleep, q2_vo, sleep, q2_gr
    q2_vo_idx = None
    vo_count = 0
    for idx, ev in enumerate(events):
        if ev == ("query", "vector_only"):
            vo_count += 1
            if vo_count == 2:
                q2_vo_idx = idx
                break
    assert q2_vo_idx is not None
    assert events[q2_vo_idx - 1][0] == "sleep"


def test_ollama_no_groq_pacing(_mock_env):
    """Ollama provider does not trigger Groq pacing sleeps."""
    settings, mock_rq, qpath = _mock_env
    settings.llm_provider = "ollama"
    sleep_calls = []

    async def _tracking_sleep(secs):
        sleep_calls.append(secs)

    with patch("evaluation.evaluator.asyncio.sleep", side_effect=_tracking_sleep):
        asyncio.run(run_evaluation(qpath))

    assert len(sleep_calls) == 0


def test_openai_no_groq_pacing(_mock_env):
    """OpenAI provider does not trigger Groq pacing sleeps."""
    settings, mock_rq, qpath = _mock_env
    settings.llm_provider = "openai"
    sleep_calls = []

    async def _tracking_sleep(secs):
        sleep_calls.append(secs)

    with patch("evaluation.evaluator.asyncio.sleep", side_effect=_tracking_sleep):
        asyncio.run(run_evaluation(qpath))

    assert len(sleep_calls) == 0


def test_evaluation_abort_error_still_raised(_mock_env):
    """EvaluationAbortError propagates even with Groq pacing enabled."""
    settings, mock_rq, qpath = _mock_env

    async def _failing_rq(question, mode="graphrag"):
        return {
            "answer": "",
            "reranked_evidence": [],
            "graph_evidence": [],
            "vector_evidence": [],
            "errors": ["Vector retrieval error: connection refused"],
        }

    mock_rq.side_effect = _failing_rq

    with pytest.raises(EvaluationAbortError):
        asyncio.run(run_evaluation(qpath))
