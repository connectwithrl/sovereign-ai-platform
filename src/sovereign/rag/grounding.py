"""Grounding evaluation.

Treats grounding as engineering, not a research artefact: every answer is scored against
the retrieved context it was supposed to be grounded in. The score is the fraction of the
answer's content sentences that have sufficient lexical support in the context. Answers
below `min_grounding` should be flagged/blocked by the caller — this is the hook for the
hallucination guardrail in a public-sector deployment.

A lexical overlap metric is deliberately simple and explainable; swap in an NLI entailment
model for stronger guarantees (the interface stays the same).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_STOP = frozenset(
    "the a an of to and or in on at for is are was were be been it this that with as by from".split()
)


def _content_terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2}


@dataclass
class SentenceGrounding:
    sentence: str
    support: float
    grounded: bool


@dataclass
class GroundingReport:
    score: float  # fraction of grounded sentences in [0, 1]
    grounded: bool
    sentences: list[SentenceGrounding]


def evaluate_grounding(
    answer: str,
    contexts: list[str],
    *,
    sentence_threshold: float = 0.6,
    min_grounding: float = 0.5,
) -> GroundingReport:
    context_terms = set()
    for c in contexts:
        context_terms |= _content_terms(c)

    sentences = [s.strip() for s in _SENT_RE.split(answer.strip()) if s.strip()]
    if not sentences:
        return GroundingReport(score=0.0, grounded=False, sentences=[])

    details: list[SentenceGrounding] = []
    grounded_count = 0
    for s in sentences:
        terms = _content_terms(s)
        if not terms:
            # No checkable content (e.g. "Here is the answer:") — treat as neutral/grounded.
            details.append(SentenceGrounding(sentence=s, support=1.0, grounded=True))
            grounded_count += 1
            continue
        support = len(terms & context_terms) / len(terms)
        is_grounded = support >= sentence_threshold
        grounded_count += int(is_grounded)
        details.append(SentenceGrounding(sentence=s, support=round(support, 3), grounded=is_grounded))

    score = grounded_count / len(sentences)
    return GroundingReport(score=round(score, 3), grounded=score >= min_grounding, sentences=details)