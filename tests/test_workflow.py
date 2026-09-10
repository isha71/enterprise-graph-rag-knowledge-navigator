def test_workflow_compiles():
    """Test that the LangGraph workflow can be imported and has correct structure."""
    from app.workflow.graph import workflow, builder
    assert workflow is not None


def test_workflow_has_nodes():
    """Test that all expected nodes exist."""
    from app.workflow.graph import builder
    # The builder should have the nodes we defined
    expected_nodes = ["analyze_query", "vector_retrieval", "graph_retrieval",
                      "merge_context", "rerank_context", "generate_answer"]
    for node_name in expected_nodes:
        assert node_name in builder.nodes, f"Missing node: {node_name}"


def test_state_schema():
    """Test that GraphRAGState has expected fields."""
    from app.workflow.state import GraphRAGState
    annotations = GraphRAGState.__annotations__
    assert "question" in annotations
    assert "query_analysis" in annotations
    assert "vector_evidence" in annotations
    assert "graph_evidence" in annotations
    assert "answer" in annotations
