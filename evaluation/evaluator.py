"""Evaluation engine for comparing vector-only vs GraphRAG."""
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from app.workflow.graph import run_query
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _normalize_for_match(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def check_answer_accuracy(answer: str, expected_terms: list[str]) -> bool:
    """Check if all expected terms appear in the answer."""
    answer_norm = _normalize_for_match(answer)
    return all(_normalize_for_match(t) in answer_norm for t in expected_terms)


def check_evidence_hit(evidence: list[dict], expected_terms: list[str]) -> bool:
    """Check if expected terms appear in retrieved evidence."""
    all_text = _normalize_for_match(
        " ".join(e.get("text", "") for e in evidence)
    )
    return all(_normalize_for_match(t) in all_text for t in expected_terms)


def path_contains_ordered_entities(
    actual_entities: list[str],
    expected_entities: list[str],
) -> bool:
    """Check if expected entities appear in order within a single path.
    
    Both lists are compared after case/whitespace normalization.
    Expected entities must appear in order but not necessarily adjacent.
    """
    if not expected_entities:
        return True
    
    norm_actual = [_normalize_for_match(e) for e in actual_entities]
    norm_expected = [_normalize_for_match(e) for e in expected_entities]
    
    idx = 0
    for actual in norm_actual:
        if idx < len(norm_expected) and norm_expected[idx] in actual:
            idx += 1
    
    return idx == len(norm_expected)


def check_graph_path_hit(graph_evidence: list[dict], expected_entities: list[str]) -> bool:
    """Check if ANY single retrieved graph path contains the expected entities in order.
    
    This verifies a genuine multi-hop traversal rather than scattered mentions
    across unrelated paths.
    """
    if not expected_entities:
        return True
    
    for ev in graph_evidence:
        path_nodes = ev.get("metadata", {}).get("path_nodes", [])
        if path_nodes:
            actual_names = [n.get("name", "") for n in path_nodes]
            if path_contains_ordered_entities(actual_names, expected_entities):
                return True
        
        text = ev.get("text", "")
        if text:
            parts = [p.strip() for p in re.split(r'-\[.*?\]->', text)]
            parts = [p for sublist in [p.split('<-') for p in parts] for p in sublist]
            parts = [re.sub(r'\[.*?\]-?\s*', '', p).strip() for p in parts]
            parts = [p for p in parts if p]
            if path_contains_ordered_entities(parts, expected_entities):
                return True
    
    return False


def calc_rate(results: list[dict], mode_key: str, field: str, category: str | None = None) -> float:
    """Calculate hit rate for a given field across results."""
    filtered = results
    if category:
        filtered = [r for r in results if r["category"] == category]
    if not filtered:
        return 0.0
    hits = sum(1 for r in filtered if r[mode_key].get(field, False))
    return round(hits / len(filtered), 4)


def calc_path_rate(results: list[dict], category: str | None = None) -> float:
    """Graph path hit rate only over questions with non-empty expected_path_entities."""
    filtered = [r for r in results if r.get("expected_path_entities")]
    if category:
        filtered = [r for r in filtered if r["category"] == category]
    if not filtered:
        return 0.0
    hits = sum(1 for r in filtered if r["graph_rag"].get("graph_path_hit", False))
    return round(hits / len(filtered), 4)


class EvaluationAbortError(Exception):
    """Raised when evaluation must abort due to infrastructure failure."""


_CRITICAL_ERROR_KEYWORDS = [
    "vector retrieval error",
    "graph retrieval error",
    "reranking error",
    "generation error",
    "query analysis error",
]


def _has_critical_errors(errors: list[str], mode: str) -> list[str]:
    """Return list of critical errors found in workflow result."""
    critical = []
    for err in errors:
        err_lower = err.lower()
        for keyword in _CRITICAL_ERROR_KEYWORDS:
            if keyword in err_lower:
                if mode == "vector_only" and "graph retrieval" in err_lower:
                    continue
                critical.append(err)
                break
    return critical


async def evaluate_question(question_data: dict, mode: str) -> dict:
    """Evaluate a single question. Raises EvaluationAbortError on infrastructure failure."""
    question = question_data["question"]
    expected_terms = question_data.get("expected_answer_terms", [])
    expected_path = question_data.get("expected_path_entities", [])
    
    result = await run_query(question, mode=mode)
    
    errors = result.get("errors", [])
    critical = _has_critical_errors(errors, mode)
    if critical:
        raise EvaluationAbortError(
            f"Infrastructure failure during {mode} evaluation of "
            f"'{question_data['id']}': {critical}"
        )
    
    answer = result.get("answer", "")
    reranked = result.get("reranked_evidence", [])
    graph_ev = result.get("graph_evidence", [])
    
    answer_correct = check_answer_accuracy(answer, expected_terms)
    evidence_hit = check_evidence_hit(reranked, expected_terms)
    graph_hit = check_graph_path_hit(graph_ev, expected_path) if mode == "graphrag" else None
    
    return {
        "question_id": question_data["id"],
        "category": question_data["category"],
        "question": question,
        "mode": mode,
        "answer": answer,
        "answer_correct": answer_correct,
        "evidence_hit": evidence_hit,
        "graph_path_hit": graph_hit,
        "vector_evidence_count": len(result.get("vector_evidence", [])),
        "graph_evidence_count": len(graph_ev),
        "reranked_evidence_count": len(reranked),
        "errors": errors,
    }


async def run_evaluation(questions_path: str = None) -> dict:
    """Run full evaluation comparing vector-only vs GraphRAG."""
    settings = get_settings()
    
    if questions_path is None:
        questions_path = str(Path(__file__).parent / "questions.json")
    
    with open(questions_path) as f:
        questions = json.load(f)
    
    print(f"Running evaluation with {len(questions)} questions")
    print("=" * 60)
    
    per_question = []
    is_groq = settings.llm_provider.lower() == "groq"
    groq_delay = settings.groq_request_delay_seconds if is_groq else 0
    
    for i, q in enumerate(questions):
        print(f"\n[{q['id']}] {q['question']}")
        
        if is_groq and i > 0:
            logger.info("Groq pacing: waiting %.0fs before next evaluation run", groq_delay)
            await asyncio.sleep(groq_delay)
        
        print("  Running vector_only...")
        vo_result = await evaluate_question(q, "vector_only")
        
        if is_groq:
            logger.info("Groq pacing: waiting %.0fs before next evaluation run", groq_delay)
            await asyncio.sleep(groq_delay)
        
        print("  Running graphrag...")
        gr_result = await evaluate_question(q, "graphrag")
        
        per_question.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_answer_terms": q.get("expected_answer_terms", []),
            "expected_path_entities": q.get("expected_path_entities", []),
            "vector_only": vo_result,
            "graph_rag": gr_result,
        })
        
        print(f"  Vector-only: answer={vo_result['answer_correct']}, evidence={vo_result['evidence_hit']}")
        print(f"  GraphRAG:    answer={gr_result['answer_correct']}, evidence={gr_result['evidence_hit']}, path={gr_result['graph_path_hit']}")
    
    categories = ["factual", "semantic", "relationship", "multi_hop"]
    
    vo_acc = calc_rate(per_question, "vector_only", "answer_correct")
    gr_acc = calc_rate(per_question, "graph_rag", "answer_correct")
    vo_ev = calc_rate(per_question, "vector_only", "evidence_hit")
    gr_ev = calc_rate(per_question, "graph_rag", "evidence_hit")
    gr_path = calc_path_rate(per_question)
    gr_mh_path = calc_path_rate(per_question, "multi_hop")
    
    vo_acc_cat = {c: calc_rate(per_question, "vector_only", "answer_correct", c) for c in categories}
    gr_acc_cat = {c: calc_rate(per_question, "graph_rag", "answer_correct", c) for c in categories}
    vo_ev_cat = {c: calc_rate(per_question, "vector_only", "evidence_hit", c) for c in categories}
    gr_ev_cat = {c: calc_rate(per_question, "graph_rag", "evidence_hit", c) for c in categories}
    gr_path_cat = {c: calc_path_rate(per_question, c) for c in categories}
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.ollama_model or settings.openai_model,
        },
        "vector_only": {
            "overall_accuracy": vo_acc,
            "category_accuracy": vo_acc_cat,
            "overall_evidence_hit": vo_ev,
            "category_evidence_hit": vo_ev_cat,
        },
        "graph_rag": {
            "overall_accuracy": gr_acc,
            "category_accuracy": gr_acc_cat,
            "overall_evidence_hit": gr_ev,
            "category_evidence_hit": gr_ev_cat,
            "graph_path_hit_rate": gr_path,
            "multi_hop_graph_path_hit_rate": gr_mh_path,
            "category_graph_path_hit": gr_path_cat,
        },
        "per_question": per_question,
    }
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "latest.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Category':<15s}  {'V-Only Acc':>10s}  {'GR Acc':>10s}  {'V-Only Ev':>10s}  {'GR Ev':>10s}  {'GR Path':>10s}")
    print("-" * 75)
    for cat in categories:
        print(f"  {cat:<13s}  {vo_acc_cat[cat]:>10.2f}  {gr_acc_cat[cat]:>10.2f}  {vo_ev_cat[cat]:>10.2f}  {gr_ev_cat[cat]:>10.2f}  {gr_path_cat[cat]:>10.2f}")
    print("-" * 75)
    print(f"  {'Overall':<13s}  {vo_acc:>10.2f}  {gr_acc:>10.2f}  {vo_ev:>10.2f}  {gr_ev:>10.2f}  {gr_path:>10.2f}")
    print(f"\n  Multi-hop graph path hit rate: {gr_mh_path:.2f}")
    print(f"\nResults saved to: {results_path}")
    
    return report
