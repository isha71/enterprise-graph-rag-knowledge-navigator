"""Tests for QdrantStore.search() using the query_points API."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.domain import EvidenceItem


def _make_scored_point(chunk_id, text, doc_id, doc_name, score, page=1, chunk_idx=0):
    pt = MagicMock()
    pt.payload = {
        "chunk_id": chunk_id,
        "text": text,
        "document_id": doc_id,
        "document_name": doc_name,
        "page_number": page,
        "chunk_index": chunk_idx,
    }
    pt.score = score
    return pt


def _build_store(points):
    """Build a QdrantStore with mocked client and embedding service."""
    with patch("app.vectorstore.qdrant_store.QdrantClient"), \
         patch("app.vectorstore.qdrant_store.EmbeddingService") as MockEmb:
        mock_emb_instance = MagicMock()
        mock_emb_instance.dimension = 384
        mock_emb_instance.embed_single.return_value = [0.1] * 384
        MockEmb.get_instance.return_value = mock_emb_instance

        from app.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore()

        response = MagicMock()
        response.points = points
        store._client.query_points = MagicMock(return_value=response)

        return store


def test_search_calls_query_points_not_search():
    """search() must use client.query_points(), not client.search()."""
    store = _build_store([])
    store.search("test query")
    store._client.query_points.assert_called_once()
    assert not hasattr(store._client, "search") or not store._client.search.called


def test_search_passes_query_vector():
    """The embedding vector must be passed as query= argument."""
    store = _build_store([])
    store.search("test query")
    call_kwargs = store._client.query_points.call_args
    assert "query" in call_kwargs.kwargs or len(call_kwargs.args) > 1
    query_val = call_kwargs.kwargs.get("query")
    assert query_val is not None
    assert len(query_val) == 384


def test_search_passes_limit():
    store = _build_store([])
    store.search("test query", top_k=5)
    call_kwargs = store._client.query_points.call_args.kwargs
    assert call_kwargs["limit"] == 5


def test_search_requests_with_payload():
    store = _build_store([])
    store.search("test query")
    call_kwargs = store._client.query_points.call_args.kwargs
    assert call_kwargs["with_payload"] is True


def test_search_reads_response_points():
    """Results must come from response.points."""
    pt = _make_scored_point("c1", "hello", "d1", "doc.md", 0.95)
    store = _build_store([pt])
    results = store.search("test query")
    assert len(results) == 1


def test_search_converts_to_evidence_items():
    pt = _make_scored_point("c1", "chunk text", "d1", "doc.md", 0.92, page=2, chunk_idx=3)
    store = _build_store([pt])
    results = store.search("test query")
    assert len(results) == 1
    ev = results[0]
    assert isinstance(ev, EvidenceItem)
    assert ev.id == "c1"
    assert ev.text == "chunk text"
    assert ev.evidence_type == "vector"
    assert ev.retrieval_score == 0.92
    assert ev.source_document_ids == ["d1"]
    assert ev.source_chunk_ids == ["c1"]
    assert ev.metadata["document_name"] == "doc.md"
    assert ev.metadata["page_number"] == 2
    assert ev.metadata["chunk_index"] == 3


def test_search_converts_numpy_array_to_list():
    """If embed_single returns a numpy array, it must be converted to list."""
    with patch("app.vectorstore.qdrant_store.QdrantClient"), \
         patch("app.vectorstore.qdrant_store.EmbeddingService") as MockEmb:
        mock_emb_instance = MagicMock()
        mock_emb_instance.dimension = 384
        mock_emb_instance.embed_single.return_value = np.zeros(384)
        MockEmb.get_instance.return_value = mock_emb_instance

        from app.vectorstore.qdrant_store import QdrantStore
        store = QdrantStore()

        response = MagicMock()
        response.points = []
        store._client.query_points = MagicMock(return_value=response)

        store.search("test query")
        query_val = store._client.query_points.call_args.kwargs["query"]
        assert isinstance(query_val, list)
