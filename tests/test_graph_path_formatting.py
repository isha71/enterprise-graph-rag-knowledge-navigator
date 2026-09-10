from app.graph.retrieval import format_graph_path, paths_to_evidence


def test_format_forward_path():
    """When traversal matches stored direction, render A -[REL]-> B."""
    path = {
        "nodes": [
            {"name": "Bob Chen", "type": "Employee"},
            {"name": "AI Platform Team", "type": "Team"},
        ],
        "relationships": [
            {"type": "MANAGES", "forward": True},
        ],
    }
    result = format_graph_path(path)
    assert result.text == "Bob Chen -[MANAGES]-> AI Platform Team"


def test_format_reverse_path():
    """Stored: AI Platform Team -[OWNS]-> Project Atlas.
    Traversal from Project Atlas visits in reverse order."""
    path = {
        "nodes": [
            {"name": "Project Atlas", "type": "Project"},
            {"name": "AI Platform Team", "type": "Team"},
        ],
        "relationships": [
            {"type": "OWNS", "forward": False},
        ],
    }
    result = format_graph_path(path)
    assert result.text == "Project Atlas <-[OWNS]- AI Platform Team"


def test_format_multi_hop_forward():
    """All relationships traversed in their stored direction."""
    path = {
        "nodes": [
            {"name": "Bob Chen", "type": "Employee"},
            {"name": "AI Platform Team", "type": "Team"},
            {"name": "Project Atlas", "type": "Project"},
            {"name": "Neo4j", "type": "Technology"},
        ],
        "relationships": [
            {"type": "MANAGES", "forward": True},
            {"type": "OWNS", "forward": True},
            {"type": "USES", "forward": True},
        ],
    }
    result = format_graph_path(path)
    assert result.text == "Bob Chen -[MANAGES]-> AI Platform Team -[OWNS]-> Project Atlas -[USES]-> Neo4j"


def test_format_mixed_direction_path():
    """Some relationships traversed forward, some backward."""
    path = {
        "nodes": [
            {"name": "Project Atlas", "type": "Project"},
            {"name": "AI Platform Team", "type": "Team"},
            {"name": "Bob Chen", "type": "Employee"},
        ],
        "relationships": [
            {"type": "OWNS", "forward": False},
            {"type": "MANAGES", "forward": False},
        ],
    }
    result = format_graph_path(path)
    assert result.text == "Project Atlas <-[OWNS]- AI Platform Team <-[MANAGES]- Bob Chen"


def test_format_default_forward_when_missing():
    """If 'forward' key is missing, default to forward arrow."""
    path = {
        "nodes": [
            {"name": "A", "type": "Employee"},
            {"name": "B", "type": "Team"},
        ],
        "relationships": [
            {"type": "MANAGES"},
        ],
    }
    result = format_graph_path(path)
    assert "->" in result.text


def test_paths_to_evidence():
    paths = [
        {
            "nodes": [
                {"name": "A", "type": "Employee", "source_chunk_ids": ["c1"], "source_document_ids": ["d1"]},
                {"name": "B", "type": "Team", "source_chunk_ids": ["c2"], "source_document_ids": ["d1"]},
            ],
            "relationships": [
                {"type": "MANAGES", "forward": True, "source_chunk_ids": ["c1"], "source_document_ids": ["d1"]},
            ],
        }
    ]
    evidence = paths_to_evidence(paths)
    assert len(evidence) == 1
    assert evidence[0].evidence_type == "graph"
    assert "d1" in evidence[0].source_document_ids


def test_deduplicate_paths():
    path = {
        "nodes": [
            {"name": "A", "type": "Employee", "source_chunk_ids": [], "source_document_ids": []},
            {"name": "B", "type": "Team", "source_chunk_ids": [], "source_document_ids": []},
        ],
        "relationships": [{"type": "MANAGES", "forward": True, "source_chunk_ids": [], "source_document_ids": []}],
    }
    evidence = paths_to_evidence([path, path])
    assert len(evidence) == 1
