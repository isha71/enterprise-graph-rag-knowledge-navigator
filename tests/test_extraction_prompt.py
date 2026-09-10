"""Tests verifying the extraction prompt contains required instructions and examples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.extraction import build_extraction_prompt

PROMPT = build_extraction_prompt("Sample text for testing.")


def test_prompt_contains_exhaustive_extraction_instruction():
    assert "extract all explicitly stated relationships" in PROMPT.lower()


def test_prompt_contains_sentence_by_sentence_instruction():
    assert "sentence by sentence" in PROMPT.lower()


def test_prompt_contains_do_not_stop_instruction():
    assert "do not stop after finding one relationship" in PROMPT.lower()


def test_prompt_contains_supported_by_definition():
    assert "SUPPORTED_BY: Technology -> Vendor" in PROMPT


def test_prompt_contains_uses_definition():
    assert "USES: Project -> Technology" in PROMPT


def test_prompt_contains_depends_on_definition():
    assert "DEPENDS_ON: Project -> Service" in PROMPT


def test_prompt_contains_manages_definition():
    assert "MANAGES: Employee -> Team" in PROMPT


def test_prompt_contains_positive_example():
    assert "Neo4j -[SUPPORTED_BY]-> GraphSphere Technologies" in PROMPT


def test_prompt_contains_positive_uses_example():
    assert "Project Atlas -[USES]-> Neo4j" in PROMPT


def test_prompt_contains_negative_anti_hallucination_example():
    assert "Do NOT extract: AI Platform Team -[USES]-> Neo4j" in PROMPT


def test_prompt_contains_negative_depends_on_example():
    assert "Do NOT extract: Project Atlas -[DEPENDS_ON]-> Neo4j" in PROMPT


def test_prompt_contains_negative_manages_example():
    assert "Do NOT extract: Alice -[MANAGES]-> Project Atlas" in PROMPT


def test_prompt_preserves_json_output_format():
    assert '"entities"' in PROMPT
    assert '"relationships"' in PROMPT
    assert '"source"' in PROMPT
    assert '"type"' in PROMPT
    assert '"target"' in PROMPT
