# 3. Pluggable backends selected by configuration

## Status

Accepted

## Context

The platform has to run in two very different places without code changes:

- **CI and the offline demo**, where there is no GPU, no model server, no API
  keys, and no external network. Tests must be deterministic and fast.
- **Production**, where serving is a self-hosted vLLM/TGI server (or AWS Bedrock),
  embeddings come from a real model, and the vector store is PostgreSQL + pgvector.

If those two environments diverge into separate code paths, they drift: the thing
that passes CI is not the thing that ships. We want one image and one code path,
with the environment-specific pieces chosen at startup.

There are three independent concerns that need swapping: LLM serving, embeddings,
and the vector store.

## Decision

Each concern is defined by a small `typing.Protocol` and constructed by a single
`build_*(settings)` factory that reads typed `pydantic-settings`
(`SOVEREIGN_`-prefixed environment variables). The offline default for each is
deterministic and dependency-free; production backends are imported lazily so the
core package installs without the heavy extras.

| Concern    | Protocol         | Factory                       | Offline default        | Production option(s)                          | Env var                        |
|------------|------------------|-------------------------------|------------------------|-----------------------------------------------|--------------------------------|
| Serving    | `ServingBackend` | `build_backend` (`serving.py`)| `EchoBackend`          | `OpenAICompatBackend` (vLLM/TGI), `BedrockBackend` | `SOVEREIGN_SERVING_BACKEND`    |
| Embeddings | `Embedder`       | `build_embedder` (`embeddings.py`)| `HashingEmbedder`  | `OpenAIEmbedder`                              | `SOVEREIGN_EMBEDDING_BACKEND`  |
| Vector store | `VectorStore`  | `build_store` (`rag/store.py`)| `InMemoryStore`        | `PgVectorStore`                              | `SOVEREIGN_DATABASE_URL`       |

- `EchoBackend` returns a deterministic extractive answer from the supplied
  context, which keeps the whole pipeline (retrieve → prompt → answer → grounding
  eval) exercisable without a real model.
- `HashingEmbedder` is a hashed bag of unigrams + bigrams projected to a fixed
  dimension and L2-normalised — deterministic and reproducible.
- `InMemoryStore` does exact cosine search over an in-memory matrix.

The application (`api.py`) and the MCP server (`mcp_server.py`) both build their
state through these same factories, so HTTP and MCP share one implementation and
one set of guarantees.

## Consequences

- The same container image runs in CI, in the demo, and in production; only
  environment variables change. No environment drift.
- Tests are hermetic and fast: no network, no GPU, no keys, fully reproducible
  vectors and answers.
- Swapping a provider is a config change, and adding a new provider is a new
  class plus a branch in the relevant `build_*` factory — the rest of the
  platform is untouched because it depends only on the protocol.
- **Operational risk to call out explicitly:** the offline defaults are *not
  semantic*. `HashingEmbedder` has no notion of meaning, `EchoBackend` does not
  reason, and `InMemoryStore` does not scale. They exist to make the pipeline
  testable, not to be served to users. A production deployment **must** set
  `SOVEREIGN_SERVING_BACKEND`, `SOVEREIGN_EMBEDDING_BACKEND`, and
  `SOVEREIGN_DATABASE_URL` to real backends. Shipping with the defaults silently
  in place would produce keyword-matched, non-semantic results. This should be
  enforced as a startup check / deployment guardrail, not left to operator memory.