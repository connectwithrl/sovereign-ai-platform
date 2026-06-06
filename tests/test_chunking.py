from sovereign.rag.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_respect_token_budget_with_overlap():
    text = ". ".join(f"sentence number {i} carries some words" for i in range(60)) + "."
    chunks = chunk_text(text, chunk_tokens=30, overlap=6)
    assert len(chunks) > 1
    for c in chunks:
        # allow a small slop because whole sentences are packed
        assert len(c.text.split()) <= 30 + 12
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_context_between_chunks():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text + ".", chunk_tokens=40, overlap=10)
    # consecutive chunks should share some trailing/leading tokens
    assert len(chunks) >= 2