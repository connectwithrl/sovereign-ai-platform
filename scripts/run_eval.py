"""Run the offline evaluation harness against the configured backends.

Builds an agent from the current settings (in-memory + offline by default, or the
production backends if SOVEREIGN_* env vars point at them), ingests the sample corpus,
and reports retrieval hit-rate, MRR, mean grounding, and substring accuracy.

Wire this into CI to catch retrieval/grounding regressions like any other test.
"""

from __future__ import annotations

import os
import sys

from sovereign.agent import RagAgent
from sovereign.config import get_settings
from sovereign.embeddings import build_embedder
from sovereign.eval import EvalCase, run_eval
from sovereign.rag.ingest import ingest_document
from sovereign.rag.store import build_store
from sovereign.serving import build_backend

sys.path.insert(0, os.path.dirname(__file__))
from sample_data import EVAL_CASES, SAMPLE_DOCS  # noqa: E402


def main() -> None:
    settings = get_settings()
    store = build_store(settings)
    embedder = build_embedder(settings)
    backend = build_backend(settings)

    for doc_id, text in SAMPLE_DOCS.items():
        ingest_document(text=text, source=doc_id, store=store, embedder=embedder, doc_id=doc_id)

    agent = RagAgent(store=store, embedder=embedder, backend=backend, settings=settings)
    cases = [EvalCase(q, doc, subs) for q, doc, subs in EVAL_CASES]
    result = run_eval(agent, cases, top_k=settings.top_k)

    print(f"Eval over {result.n} cases  (serving={settings.serving_backend}, "
          f"embeddings={settings.embedding_backend})")
    print(f"  retrieval hit-rate : {result.hit_rate:.3f}")
    print(f"  retrieval MRR      : {result.mrr:.3f}")
    print(f"  mean grounding     : {result.grounding:.3f}")
    print(f"  substring accuracy : {result.substring_rate:.3f}")

    # Non-zero exit if retrieval regresses below a floor — usable as a CI gate.
    floor = float(os.environ.get("EVAL_HITRATE_FLOOR", "0.75"))
    if result.hit_rate < floor:
        print(f"FAIL: hit-rate {result.hit_rate:.3f} < floor {floor:.3f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()