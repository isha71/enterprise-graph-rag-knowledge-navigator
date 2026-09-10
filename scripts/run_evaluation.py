"""Run the evaluation pipeline."""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.logging import setup_logging
from evaluation.evaluator import run_evaluation

setup_logging("INFO")


async def main():
    await run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
