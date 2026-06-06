import pytest

from sovereign.embeddings import build_embedder
from sovereign.rag.grounding import evaluate_grounding
from sovereign.rag.ingest import ingest_document
from sovereign.rag.retrieve import retrieve
from sovereign.rag.store import InMemoryStore

from .conftest import SAMPLE_DOCS


@pytest.fixture
def populated(settings):
    store = InMemoryStore()
    embedder = build_embedder(settings)
    for source, text in SAMPLE_DOCS.items():
        ingest_document(text=text, source=source, store=store, embedder=embedder, doc_id=source)
    return store, embedder


def test_ingest_is_idempotent(settings):
    store = InMemoryStore()
    embedder = build_embedder(settings)
    r1 = ingest_document(text=SAMPLE_DOCS["leave-policy"], source="leave", store=store,
                         embedder=embedder, doc_id="leave")
    count_after_first = store.count()
    r2 = ingest_document(text=SAMPLE_DOCS["leave-policy"], source="leave", store=store,
                         embedder=embedder, doc_id="leave")
    assert r1.chunks == r2.chunks
    assert r2.replaced == count_after_first
    assert store.count() == count_after_first  # no duplication


def test_retrieval_finds_right_document(populated, settings):
    store, embedder = populated
    hits = retrieve(query="how many days of annual leave do employees get?", store=store,
                    embedder=embedder, top_k=3)
    assert hits
    assert hits[0].doc_id == "leave-policy"


def test_retrieval_separates_topics(populated, settings):
    store, embedder = populated
    hits = retrieve(query="can confidential data go to external providers?", store=store,
                    embedder=embedder, top_k=3)
    assert hits[0].doc_id == "data-classification"


def test_delete_document(populated):
    store, _ = populated
    before = store.count()
    removed = store.delete_document("procurement")
    assert removed > 0
    assert store.count() == before - removed


def test_grounding_high_for_supported_answer():
    contexts = ["Employees accrue thirty calendar days of paid annual leave each year."]
    report = evaluate_grounding("Employees accrue thirty days of paid annual leave each year.", contexts)
    assert report.grounded
    assert report.score >= 0.5


def test_grounding_low_for_hallucination():
    contexts = ["Employees accrue thirty calendar days of paid annual leave each year."]
    report = evaluate_grounding(
        "The company stock price doubled after the quarterly earnings announcement.", contexts
    )
    assert not report.grounded