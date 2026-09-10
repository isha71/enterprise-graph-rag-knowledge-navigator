"""LangGraph workflow definition."""
from langgraph.graph import StateGraph, START, END
from app.workflow.state import GraphRAGState
from app.workflow.nodes import (
    analyze_query,
    vector_retrieval,
    graph_retrieval,
    merge_context,
    rerank_context,
    generate_answer,
)

# Build the workflow
builder = StateGraph(GraphRAGState)

builder.add_node("analyze_query", analyze_query)
builder.add_node("vector_retrieval", vector_retrieval)
builder.add_node("graph_retrieval", graph_retrieval)
builder.add_node("merge_context", merge_context)
builder.add_node("rerank_context", rerank_context)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "analyze_query")
builder.add_edge("analyze_query", "vector_retrieval")
builder.add_edge("vector_retrieval", "graph_retrieval")
builder.add_edge("graph_retrieval", "merge_context")
builder.add_edge("merge_context", "rerank_context")
builder.add_edge("rerank_context", "generate_answer")
builder.add_edge("generate_answer", END)

workflow = builder.compile()


async def run_query(question: str, mode: str = "graphrag") -> dict:
    """Execute the full GraphRAG query pipeline."""
    initial_state = {
        "question": question,
        "mode": mode,
        "errors": [],
    }
    result = await workflow.ainvoke(initial_state)
    return result
