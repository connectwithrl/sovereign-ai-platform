# 4. Two-stage retrieval: vector recall then lexical rerank

## Status

Accepted

## Context

Pure vector search is good at semantic similarity but routinely misses exact
matches — a policy number, a form ID, a regulation citation, a specific Arabic or
English term — that are common and load-bearing in government and policy
documents. A query for "Form HR-7" or "Circular 12/2023" should not lose to a
chunk that is merely *topically* close.

The standard fix is a reranking stage. A learned cross-encoder (BGE reranker,
Cohere Rerank) is the strongest option, but it is heavier to operate: another
model to host, GPU/latency budget to spend, and a dependency to certify inside the
sovereign boundary. We want the precision win without committing to that operating
cost up front.

## Decision

Use two-stage retrieval in `retrieve()` (`src/sovereign/rag/retrieve.py`):

1. **Recall.** Embed the query and pull `rerank_candidates` nearest chunks from
   the vector store (a wider net than the final `top_k`).
2. **Rerank.** Score each candidate's lexical term-overlap with the query
   (`_lexical_overlap`) and blend it with the vector score:

   ```
   final = alpha * vector_score + (1 - alpha) * lexical_overlap
   ```

   with `alpha = 0.5` by default. Re-sort by `final` and return the top `top_k`.
   Both the per-result `vector_score` and the blended `score` are returned so the
   blend is observable.

Reranking is on by default (`SOVEREIGN_RERANK`); when off, the vector score is
used directly. The lexical reranker is deliberately a cheap, dependency-free
stand-in. The two-stage shape — recall a wider candidate set, then reorder it — is
exactly the integration point a cross-encoder slots into, so swapping in a
BGE/Cohere reranker later is a localised change behind the same call.

## Consequences

- Keyword- and ID-heavy queries get materially better precision at near-zero
  added cost and with no extra infrastructure.
- The candidate-recall / rerank structure is already in place, so adopting a
  cross-encoder is a drop-in upgrade rather than a redesign.
- **Honest limitation:** lexical overlap is a bag-of-words heuristic, not
  state-of-the-art reranking. It does not capture paraphrase, synonymy, or
  cross-lingual matches, and the fixed `alpha` is a global blend, not a learned
  one. A cross-encoder earns its operating cost precisely where this stand-in is
  weakest: nuanced semantic ranking, paraphrased queries, and EN/AR cross-lingual
  retrieval. The recommendation is to measure with the eval harness (hit-rate /
  MRR) and adopt the heavier reranker when the numbers justify the added latency
  and operational surface.