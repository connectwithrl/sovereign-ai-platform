from sovereign.agent import RagAgent
from sovereign.embeddings import build_embedder
from sovereign.eval import EvalCase, run_eval
from sovereign.rag.ingest import ingest_document
from sovereign.rag.store import InMemoryStore
from sovereign.serving import build_backend

from .conftest import SAMPLE_DOCS


def _agent(settings):
    store = InMemoryStore()
    embedder = build_embedder(settings)
    for source, text in SAMPLE_DOCS.items():
        ingest_document(text=text, source=source, store=store, embedder=embedder, doc_id=source)
    backend = build_backend(settings)
    return RagAgent(store=store, embedder=embedder, backend=backend, settings=settings)


def test_agent_answer_is_grounded(settings):
    agent = _agent(settings)
    ans = agent.answer("how many days of annual leave are accrued each year?")
    assert ans.citations
    assert ans.grounding.score > 0  # echo backend answers extractively from context
    assert "thirty" in ans.text.lower()


def test_eval_harness_reports_metrics(settings):
    agent = _agent(settings)
    cases = [
        EvalCase("how much annual leave do employees accrue?", "leave-policy", ["thirty"]),
        EvalCase("can secret data leave the sovereign cloud?", "data-classification"),
        EvalCase("what approval is needed for large purchases?", "procurement"),
    ]
    result = run_eval(agent, cases, top_k=3)
    assert result.n == 3
    assert result.hit_rate >= 0.66  # at least 2/3 retrieved correctly
    assert 0.0 <= result.mrr <= 1.0