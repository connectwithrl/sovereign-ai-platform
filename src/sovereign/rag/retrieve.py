"""Retrieval with optional lexical reranking.

Two-stage retrieval: (1) vector recall pulls `rerank_candidates` nearest chunks; (2) a
lexical reranker reorders them by term-overlap with the query and blends the lexical and
vector scores. This is a cheap, dependency-free stand-in for a cross-encoder reranker
(swap in a BGE/Cohere reranker in production) that still demonstrably improves precision
on keyword-heavy queries. See docs/adr/0004-two-stage-retrieval.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sovereign.embeddings import Embedder
from sovereign.rag.store import ScoredRecord, VectorStore

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Retrieved:
    text: str
    source: str
    doc_id: str
    ordinal: int
    vector_score: float
    score: float  # final score after reranking


def _lexical_overlap(query: str, text: str) -> float:
    q = set(_WORD_RE.findall(query.lower()))
    if not q:
        return 0.0
    d = set(_WORD_RE.findall(text.lower()))
    return len(q & d) / len(q)


def retrieve(
    *,
    query: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int = 5,
    rerank: bool = True,
    rerank_candidates: int = 20,
    alpha: float = 0.5,
) -> list[Retrieved]:
    q_vec = embedder.embed([query])[0]
    candidates: list[ScoredRecord] = store.query(q_vec, top_k=rerank_candidates if rerank else top_k)
    results: list[Retrieved] = []
    for sr in candidates:
        lex = _lexical_overlap(query, sr.record.text)
        final = alpha * sr.score + (1 - alpha) * lex if rerank else sr.score
        results.append(
            Retrieved(
                text=sr.record.text,
                source=sr.record.metadata.get("source", sr.record.doc_id),
                doc_id=sr.record.doc_id,
                ordinal=sr.record.ordinal,
                vector_score=sr.score,
                score=final,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]