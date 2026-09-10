"""Fast retrieval-focused evaluation.

Runs retrieval pipelines (vector + reranker, and full GraphRAG retrieval)
WITHOUT final LLM answer generation.  The only LLM call per GraphRAG
question is query analysis.

This drastically reduces Groq free-tier API usage while producing real,
reproducible retrieval metrics.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from evaluation.evaluator import (
    check_evidence_hit,
    check_graph_path_hit,
    calc_rate,
    calc_path_rate,
    EvaluationAbortError,
    _has_critical_errors,
)

logger = get_logger(__name__)

RETRIEVAL_EVAL_QUESTION_IDS = [
    "f_001",
    "r_001",
    "mh_001",
    "mh_002",
]


def select_retrieval_questions(all_questions: list[dict]) -> list[dict]:
    """Select the 4 representative questions for fast retrieval evaluation."""
    id_set = set(RETRIEVAL_EVAL_QUESTION_IDS)
    selected = [q for q in all_questions if q["id"] in id_set]
    ordered = sorted(selected, key=lambda q: RETRIEVAL_EVAL_QUESTION_IDS.index(q["id"]))
    return ordered


async def _run_vector_only_retrieval(question: str) -> dict:
    """Run vector retrieval + reranking without LLM generation."""
    from app.workflow.nodes import vector_retrieval, merge_context, rerank_context

    state = {"question": question, "mode": "vector_only", "errors": []}

    vr = await vector_retrieval(state)
    state.update(vr)

    state["graph_evidence"] = []
    mc = await merge_context(state)
    state.update(mc)

    rr = await rerank_context(state)
    state.update(rr)

    return state


async def _run_graphrag_retrieval(question: str) -> dict:
    """Run query analysis + vector + graph retrieval + reranking without LLM generation."""
    from app.workflow.nodes import (
        analyze_query,
        vector_retrieval,
        graph_retrieval,
        merge_context,
        rerank_context,
    )

    state = {"question": question, "mode": "graphrag", "errors": []}

    qa = await analyze_query(state)
    state.update(qa)

    vr = await vector_retrieval(state)
    state.update(vr)

    gr = await graph_retrieval(state)
    state.update(gr)

    mc = await merge_context(state)
    state.update(mc)

    rr = await rerank_context(state)
    state.update(rr)

    return state


async def evaluate_retrieval_question(
    question_data: dict, mode: str,
) -> dict:
    """Evaluate retrieval quality for a single question (no generation)."""
    question = question_data["question"]
    expected_terms = question_data.get("expected_answer_terms", [])
    expected_path = question_data.get("expected_path_entities", [])

    if mode == "vector_only":
        result = await _run_vector_only_retrieval(question)
    else:
        result = await _run_graphrag_retrieval(question)

    errors = result.get("errors", [])
    critical = _has_critical_errors(errors, mode)
    if critical:
        raise EvaluationAbortError(
            f"Infrastructure failure during {mode} retrieval evaluation of "
            f"'{question_data['id']}': {critical}"
        )

    reranked = result.get("reranked_evidence", [])
    graph_ev = result.get("graph_evidence", [])

    evidence_hit = check_evidence_hit(reranked, expected_terms)
    graph_hit = check_graph_path_hit(graph_ev, expected_path) if mode == "graphrag" else None

    return {
        "question_id": question_data["id"],
        "category": question_data["category"],
        "question": question,
        "mode": mode,
        "evidence_hit": evidence_hit,
        "graph_path_hit": graph_hit,
        "vector_evidence_count": len(result.get("vector_evidence", [])),
        "graph_evidence_count": len(graph_ev),
        "reranked_evidence_count": len(reranked),
        "errors": errors,
    }


async def run_retrieval_evaluation(questions_path: str | None = None) -> dict:
    """Run fast retrieval-focused evaluation (no LLM generation)."""
    settings = get_settings()

    if questions_path is None:
        questions_path = str(Path(__file__).parent / "questions.json")

    with open(questions_path) as f:
        all_questions = json.load(f)

    questions = select_retrieval_questions(all_questions)

    is_groq = settings.llm_provider.lower() == "groq"
    groq_delay = settings.groq_request_delay_seconds if is_groq else 0

    print(f"Running fast retrieval evaluation with {len(questions)} questions")
    print("=" * 60)

    per_question: list[dict] = []

    for i, q in enumerate(questions):
        print(f"\n[{q['id']}] {q['question']}")

        print("  Running vector_only retrieval...")
        vo_result = await evaluate_retrieval_question(q, "vector_only")

        if is_groq:
            logger.info("Groq pacing: waiting %.0fs before next evaluation run", groq_delay)
            await asyncio.sleep(groq_delay)

        print("  Running graphrag retrieval...")
        gr_result = await evaluate_retrieval_question(q, "graphrag")

        if is_groq and i < len(questions) - 1:
            logger.info("Groq pacing: waiting %.0fs before next evaluation run", groq_delay)
            await asyncio.sleep(groq_delay)

        per_question.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_answer_terms": q.get("expected_answer_terms", []),
            "expected_path_entities": q.get("expected_path_entities", []),
            "vector_only": vo_result,
            "graph_rag": gr_result,
        })

        print(f"  Vector-only: evidence_hit={vo_result['evidence_hit']}")
        print(f"  GraphRAG:    evidence_hit={gr_result['evidence_hit']}, graph_path_hit={gr_result['graph_path_hit']}")

    # Compute aggregate metrics
    vo_ev = calc_rate(per_question, "vector_only", "evidence_hit")
    gr_ev = calc_rate(per_question, "graph_rag", "evidence_hit")
    vo_mh_ev = calc_rate(per_question, "vector_only", "evidence_hit", "multi_hop")
    gr_mh_ev = calc_rate(per_question, "graph_rag", "evidence_hit", "multi_hop")
    gr_mh_path = calc_path_rate(per_question, "multi_hop")

    report = {
        "evaluation_type": "retrieval",
        "timestamp": datetime.utcnow().isoformat(),
        "question_count": len(questions),
        "configuration": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "llm_provider": settings.llm_provider,
        },
        "vector_only": {
            "overall_evidence_hit_rate": vo_ev,
            "multi_hop_evidence_hit_rate": vo_mh_ev,
        },
        "graph_rag": {
            "overall_evidence_hit_rate": gr_ev,
            "multi_hop_evidence_hit_rate": gr_mh_ev,
            "multi_hop_graph_path_hit_rate": gr_mh_path,
        },
        "per_question": per_question,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "retrieval_latest.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Vector-only evidence hit rate:          {vo_ev:.2f}")
    print(f"  Vector-only multi-hop evidence hit rate: {vo_mh_ev:.2f}")
    print(f"\n  GraphRAG evidence hit rate:              {gr_ev:.2f}")
    print(f"  GraphRAG multi-hop evidence hit rate:    {gr_mh_ev:.2f}")
    print(f"  GraphRAG multi-hop graph path hit rate:  {gr_mh_path:.2f}")
    print(f"\nResults saved to: {results_path}")

    return report


async def run_single_question_evaluation(
    question_id: str, questions_path: str | None = None,
) -> dict:
    """Run retrieval evaluation for a single question by ID."""
    settings = get_settings()

    if question_id not in RETRIEVAL_EVAL_QUESTION_IDS:
        valid = ", ".join(RETRIEVAL_EVAL_QUESTION_IDS)
        raise ValueError(
            f"Question '{question_id}' is not in the lightweight evaluation set. "
            f"Valid IDs: {valid}"
        )

    if questions_path is None:
        questions_path = str(Path(__file__).parent / "questions.json")

    with open(questions_path) as f:
        all_questions = json.load(f)

    question_data = next(q for q in all_questions if q["id"] == question_id)

    is_groq = settings.llm_provider.lower() == "groq"
    groq_delay = settings.groq_request_delay_seconds if is_groq else 0

    print(f"Running single-question retrieval evaluation: {question_id}")
    print("=" * 60)
    print(f"\n[{question_data['id']}] {question_data['question']}")

    print("  Running vector_only retrieval...")
    vo_result = await evaluate_retrieval_question(question_data, "vector_only")

    if is_groq:
        logger.info("Groq pacing: waiting %.0fs before next evaluation run", groq_delay)
        await asyncio.sleep(groq_delay)

    print("  Running graphrag retrieval...")
    gr_result = await evaluate_retrieval_question(question_data, "graphrag")

    per_question = {
        "id": question_data["id"],
        "category": question_data["category"],
        "question": question_data["question"],
        "expected_answer_terms": question_data.get("expected_answer_terms", []),
        "expected_path_entities": question_data.get("expected_path_entities", []),
        "vector_only": vo_result,
        "graph_rag": gr_result,
    }

    print(f"  Vector-only: evidence_hit={vo_result['evidence_hit']}")
    print(f"  GraphRAG:    evidence_hit={gr_result['evidence_hit']}, graph_path_hit={gr_result['graph_path_hit']}")

    report = {
        "evaluation_type": "retrieval_single_question",
        "timestamp": datetime.utcnow().isoformat(),
        "question_count": 1,
        "question_id": question_id,
        "configuration": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "llm_provider": settings.llm_provider,
        },
        "per_question": [per_question],
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"retrieval_{question_id}.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return report
