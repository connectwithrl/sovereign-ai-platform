# 5. Grounding as a runtime guardrail and an eval-in-CI discipline

## Status

Accepted

## Context

In a public-sector deployment, a hallucinated answer is not a quality nuisance —
it is a liability. If a senior official reads an answer aloud in a live meeting,
or acts on it, an answer that is fluent but unsupported by the source documents is
worse than no answer at all. We need hallucination risk to be *measured* and
*controllable*, not assumed away.

Treating grounding as a research metric computed offline once is not enough. It
has to be a property of every answer at runtime and a regression gate in CI.

## Decision

Treat grounding as engineering: score every answer against the context it was
supposed to be grounded in, surface that score, and let the caller act on it.

- **Per-answer scoring.** `evaluate_grounding` (`src/sovereign/rag/grounding.py`)
  splits the answer into sentences and, for each sentence with checkable content,
  computes the fraction of its content terms (stop-words removed) that appear in
  the retrieved context. The grounding score is the fraction of sentences with
  sufficient support; the report also returns the per-sentence breakdown so a
  low score is explainable.
- **Runtime guardrail.** `answer_question` (`agent.py`) calls
  `evaluate_grounding` on every answer. When `block_ungrounded` is set, an answer
  below the grounding threshold is replaced with an explicit refusal rather than
  returned. The same flag is exposed on the `POST /v1/chat` API.
- **Observability.** The grounding score is emitted as a Prometheus histogram
  (`sovereign_grounding_score`) and blocked answers as a counter
  (`sovereign_grounding_blocked_total`), so hallucination risk is a dashboardable
  SLO, not an anecdote.
- **Eval in CI.** The offline eval harness (`eval.py`) runs labelled cases
  through the live pipeline and reports mean grounding alongside retrieval
  hit-rate and MRR, so a regression in grounding fails like a failing test.

## Consequences

- Hallucination risk becomes a measurable, monitorable, and enforceable property
  of the system rather than a hope.
- The refusal path gives a deployment a hard "don't answer when unsure" lever for
  high-stakes use.
- **Honest limitation:** lexical content-term support is a first approximation. It
  rewards answers that reuse the source's words and can be fooled by an answer
  that lexically echoes the context while distorting its meaning, or that
  paraphrases a correct, supported claim in different words. The upgrade path is a
  natural-language-inference (entailment) model that checks whether the context
  *entails* each sentence. Because grounding lives behind a single function with a
  stable report shape, that swap does not ripple through the guardrail, the
  metrics, or the eval harness.