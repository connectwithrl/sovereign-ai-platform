import pytest
from fastapi.testclient import TestClient

from sovereign.api import build_app
from sovereign.config import Settings

from .conftest import SAMPLE_DOCS


@pytest.fixture
def client():
    settings = Settings(serving_backend="echo", embedding_backend="hashing",
                        embedding_dim=256, database_url=None, top_k=3)
    app = build_app(settings)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "sovereign_requests_total" in r.text


def test_chat_on_empty_kb_returns_409(client):
    r = client.post("/v1/chat", json={"question": "anything?"})
    assert r.status_code == 409


def test_ingest_search_chat_flow(client):
    for source, text in SAMPLE_DOCS.items():
        r = client.post("/v1/ingest", json={"text": text, "source": source, "doc_id": source})
        assert r.status_code == 200
        assert r.json()["chunks"] >= 1

    r = client.post("/v1/search", json={"query": "annual leave days", "top_k": 3})
    assert r.status_code == 200
    assert r.json()[0]["doc_id"] == "leave-policy"

    r = client.post("/v1/chat", json={"question": "how many annual leave days are accrued?"})
    assert r.status_code == 200
    body = r.json()
    assert "thirty" in body["answer"].lower()
    assert body["citations"]
    assert 0.0 <= body["grounding_score"] <= 1.0


def test_delete_document_endpoint(client):
    client.post("/v1/ingest", json={"text": SAMPLE_DOCS["procurement"], "source": "p", "doc_id": "p"})
    r = client.request("DELETE", "/v1/documents/p")
    assert r.status_code == 200
    assert r.json()["removed_chunks"] >= 1