"""Offline evaluation harness.

Treats evaluation as engineering: a small labelled dataset (question -> expected source
doc_id, optional expected substrings) is run through the live retrieval + answer pipeline
and scored for retrieval hit-rate, MRR, and grounding. Wire this into CI to catch
regressions in retrieval quality the same way you'd catch a failing unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sovereign.agent import RagAgent


@dataclass
class EvalCase:
    question: str
    expected_doc_id: str
    expected_substrings: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    n: int
    hit_rate: float  # expected doc in top-k
    mrr: float  # mean reciprocal rank of expected doc
    grounding: float  # mean grounding score
    substring_rate: float  # answers containing all expected substrings


def run_eval(agent: RagAgent, cases: list[EvalCase], top_k: int = 5) -> EvalResult:
    if not cases:
        return EvalResult(0, 0.0, 0.0, 0.0, 0.0)
    hits = 0
    rr_sum = 0.0
    grounding_sum = 0.0
    substr_hits = 0
    for case in cases:
        retrieved = agent.search(case.question, top_k=top_k)
        rank = next(
            (i + 1 for i, r in enumerate(retrieved) if r.doc_id == case.expected_doc_id), None
        )
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        ans = agent.answer(case.question)
        grounding_sum += ans.grounding.score
        if case.expected_substrings and all(
            s.lower() in ans.text.lower() for s in case.expected_substrings
        ):
            substr_hits += 1
    n = len(cases)
    return EvalResult(
        n=n,
        hit_rate=round(hits / n, 3),
        mrr=round(rr_sum / n, 3),
        grounding=round(grounding_sum / n, 3),
        substring_rate=round(substr_hits / n, 3) if any(c.expected_substrings for c in cases) else 0.0,
    )