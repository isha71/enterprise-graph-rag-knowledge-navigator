from app.models.domain import EvidenceItem
from app.retrieval.fusion import merge_evidence


def test_merge_unique():
    v = [EvidenceItem(id="v1", text="chunk about atlas", evidence_type="vector")]
    g = [EvidenceItem(id="g1", text="Bob -[MANAGES]-> Team", evidence_type="graph")]
    merged = merge_evidence(v, g)
    assert len(merged) == 2


def test_merge_deduplicates():
    v = [EvidenceItem(id="v1", text="Same text", evidence_type="vector")]
    g = [EvidenceItem(id="g1", text="Same text", evidence_type="graph")]
    merged = merge_evidence(v, g)
    assert len(merged) == 1


def test_merge_empty():
    merged = merge_evidence([], [])
    assert len(merged) == 0


def test_merge_preserves_order():
    v = [
        EvidenceItem(id="v1", text="first", evidence_type="vector"),
        EvidenceItem(id="v2", text="second", evidence_type="vector"),
    ]
    g = [EvidenceItem(id="g1", text="third", evidence_type="graph")]
    merged = merge_evidence(v, g)
    assert merged[0].id == "v1"
    assert merged[1].id == "v2"
    assert merged[2].id == "g1"


def test_merge_case_insensitive_dedup():
    v = [EvidenceItem(id="v1", text="Hello World", evidence_type="vector")]
    g = [EvidenceItem(id="g1", text="hello world", evidence_type="graph")]
    merged = merge_evidence(v, g)
    assert len(merged) == 1
