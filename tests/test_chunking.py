from app.ingestion.chunker import chunk_text, compute_chunk_id


def test_basic_chunking():
    text = "Hello world. " * 200  # ~2600 chars
    chunks = chunk_text(text, "doc1", "test.md", chunk_size=1000, chunk_overlap=180)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 1000


def test_chunk_metadata():
    text = "Test content " * 100
    chunks = chunk_text(text, "doc1", "test.md", page_number=1, chunk_size=500, chunk_overlap=100)
    assert all(c.document_id == "doc1" for c in chunks)
    assert all(c.document_name == "test.md" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)
    assert chunks[0].chunk_index == 0


def test_chunk_overlap():
    text = "A" * 500 + "B" * 500 + "C" * 500
    chunks = chunk_text(text, "doc1", "test.md", chunk_size=600, chunk_overlap=100)
    # With overlap, chunks should share some content
    if len(chunks) > 1:
        end_of_first = chunks[0].text[-100:]
        start_of_second = chunks[1].text[:100]
        assert end_of_first == start_of_second


def test_deterministic_chunk_ids():
    text = "Test content for chunking"
    chunks1 = chunk_text(text, "doc1", "test.md", chunk_size=1000, chunk_overlap=180)
    chunks2 = chunk_text(text, "doc1", "test.md", chunk_size=1000, chunk_overlap=180)
    assert chunks1[0].chunk_id == chunks2[0].chunk_id


def test_small_text():
    text = "Short text"
    chunks = chunk_text(text, "doc1", "test.md", chunk_size=1000, chunk_overlap=180)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text"


def test_empty_text():
    text = ""
    chunks = chunk_text(text, "doc1", "test.md", chunk_size=1000, chunk_overlap=180)
    # Either 0 chunks or 1 empty chunk is acceptable
    assert len(chunks) <= 1


def test_chunk_id_determinism():
    id1 = compute_chunk_id("doc1", 1, 0, "hello")
    id2 = compute_chunk_id("doc1", 1, 0, "hello")
    assert id1 == id2
    assert len(id1) == 24
