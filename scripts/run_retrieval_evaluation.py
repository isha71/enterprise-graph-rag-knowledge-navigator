"""Run the fast retrieval-focused evaluation pipeline (no LLM generation).

Usage:
    python scripts/run_retrieval_evaluation.py              # all 4 questions
    python scripts/run_retrieval_evaluation.py --question mh_002  # single question
"""
import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.logging import setup_logging
from evaluation.retrieval_evaluator import (
    run_retrieval_evaluation,
    run_single_question_evaluation,
    RETRIEVAL_EVAL_QUESTION_IDS,
)

setup_logging("INFO")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast retrieval-focused evaluation (no LLM generation).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        metavar="ID",
        help=f"Run only the specified question. Valid IDs: {', '.join(RETRIEVAL_EVAL_QUESTION_IDS)}",
    )
    return parser.parse_args(argv)


async def main(args: argparse.Namespace):
    if args.question:
        await run_single_question_evaluation(args.question)
    else:
        await run_retrieval_evaluation()


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.run(main(parsed))
