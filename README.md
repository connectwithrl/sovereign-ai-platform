# Sovereign AI Platform

A self-hostable **RAG + agent serving platform** for regulated, data-sovereign environments
(built with a bilingual EN/AR public-sector context in mind). It packages the platform layer
that turns an LLM into a dependable product: an **inference gateway**, a **retrieval pipeline**
on PostgreSQL + pgvector, an **agent/MCP** tool surface, a **grounding guardrail**, and
**first-class observability** — all behind small interfaces so the same image runs offline in
CI or in production by changing environment variables only.

> Built AI-native, with Claude Code as part of the workflow.

---

## Why it's built this way

The hard part of applied AI is not the demo — it's everything around the model: serving it
within latency/throughput and cost budgets, keeping retrieval current and correct, proving
answers are grounded, and operating all of it inside data-sovereignty constraints where
confidential data can never leave the boundary.

Three design choices carry most of the weight (full reasoning in [`docs/adr/`](docs/adr)):

- **Pluggable backends behind interfaces** ([ADR-0003](docs/adr/0003-pluggable-backends.md)) — serving, embeddings, and the vector store are each a `Protocol` chosen by config. Offline defaults (a deterministic hashing embedder, an extractive *echo* LLM, an in-memory store) make tests and demos hermetic with **no GPU and no API keys**; production swaps in vLLM/Bedrock + pgvector via env vars. One image, no environment drift.
- **PostgreSQL + pgvector over a dedicated vector DB** ([ADR-0002](docs/adr/0002-pgvector-over-dedicated-vector-db.md)) — fewer moving parts to certify and operate inside a sovereign boundary, transactional consistency between documents and vectors, HNSW cosine index. The `VectorStore` interface keeps a dedicated ANN service available if scale demands it.
- **Grounding as a runtime guardrail, not a research artefact** ([ADR-0005](docs/adr/0005-grounding-as-guardrail.md)) — every answer is scored against the context it was supposed to use; `block_ungrounded` can refuse low-confidence answers; the score is a Prometheus metric and a CI eval assertion.

## Architecture

```mermaid
flowchart LR
    subgraph Clients
        H[HTTP client]
        M[MCP client<br/>Claude Code / agent]
    end
    H --> API[FastAPI<br/>/v1/ingest /v1/search /v1/chat]
    M --> MCP[MCP server<br/>search / answer / ingest tools]
    API --> AG[RAG pipeline]
    MCP --> AG
    AG --> EMB[Embedder<br/>hashing | OpenAI]
    AG --> VS[(Vector store<br/>in-memory | pgvector)]
    AG --> SRV[Serving backend<br/>echo | vLLM/OpenAI | Bedrock]
    AG --> GRD[Grounding eval / guardrail]
    API -. /metrics .-> PROM[Prometheus]
    PROM --> GRAF[Grafana]
    API -. OTLP .-> OTEL[OpenTelemetry]
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for request flows, the backend/config matrix, and the
scaling path.

---

## Quickstart (offline, no GPU, no keys)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # 19 tests, fully offline
sovereign-api                   # serves on http://localhost:8000
```

Drive it:

```bash
# 1) ingest a document
curl -s localhost:8000/v1/ingest -H 'content-type: application/json' -d '{
  "text": "Full-time employees accrue thirty calendar days of paid annual leave each year. Secret data must remain within the sovereign cloud boundary and may never be sent to external model providers.",
  "source": "hr-handbook", "doc_id": "hr"
}'

# 2) semantic search (vector recall + lexical rerank)
curl -s localhost:8000/v1/search -H 'content-type: application/json' \
  -d '{"query": "annual leave days", "top_k": 3}'

# 3) grounded answer with citations + grounding score
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"question": "How many annual leave days do employees get?"}'
```

The default `echo` backend answers *extractively from retrieved context*, so the full pipeline
(retrieve → rerank → prompt → answer → grounding eval) is exercisable without a model server.
Point at a real model by setting `SOVEREIGN_SERVING_BACKEND=openai` and
`SOVEREIGN_OPENAI_BASE_URL` at a local vLLM/TGI server (or `bedrock`).

## Full stack with Docker Compose (pgvector + Prometheus + Grafana)

```bash
docker compose up --build
# API        -> http://localhost:8000
# Prometheus -> http://localhost:9090
# Grafana    -> http://localhost:3000  (anonymous; "Sovereign AI Platform — Overview")
python scripts/seed.py            # load sample policy docs
python scripts/run_eval.py        # offline retrieval/grounding eval
```

This brings up the API against **PostgreSQL + pgvector** (the production retrieval path) with the
observability stack wired in.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | liveness/readiness + document count |
| `GET` | `/metrics` | Prometheus exposition |
| `POST` | `/v1/ingest` | chunk → embed → upsert a document (idempotent per `doc_id`) |
| `POST` | `/v1/search` | two-stage retrieval (vector recall + lexical rerank) |
| `POST` | `/v1/chat` | grounded answer with citations, grounding score, token usage |
| `DELETE` | `/v1/documents/{doc_id}` | remove a document's chunks |

## MCP server

The same retrieval/answer primitives are re-exported as **Model Context Protocol** tools
(`search_documents`, `answer_question`, `ingest_text`) so any MCP client — Claude Code, Claude
Desktop, a Pydantic AI agent — can drive the knowledge base. One implementation, one set of
guarantees, two transports (HTTP + MCP).

```bash
pip install -e ".[agent]"
sovereign-mcp                     # stdio transport
```

## Configuration

All config is environment-driven (`SOVEREIGN_` prefix; see [`.env.example`](.env.example)).

| Concern | Offline default | Production option | Env var |
|---|---|---|---|
| Serving | `echo` (extractive) | self-hosted vLLM/TGI, or Bedrock | `SOVEREIGN_SERVING_BACKEND` |
| Embeddings | `hashing` (deterministic) | OpenAI-compatible | `SOVEREIGN_EMBEDDING_BACKEND` |
| Vector store | in-memory | PostgreSQL + pgvector | `SOVEREIGN_DATABASE_URL` |
| Tracing | off | OTLP exporter | `SOVEREIGN_OTEL_ENABLED` |

## Observability

LLM platforms need more than RED metrics. Exposed at `/metrics` and dashboarded in Grafana:
token throughput (`sovereign_llm_tokens_total`), output-length distribution
(`sovereign_llm_output_tokens`), LLM and end-to-end latency histograms, retrieval top-score
distribution, **grounding score**, and **grounding-blocked** count — the signals you actually
build SLOs and alerts on. OpenTelemetry tracing is opt-in.

## Deployment

- **Docker**: multi-stage-free slim image (`python:3.12-slim`), non-root, healthcheck on `/healthz`.
- **Kubernetes**: Helm chart in [`deploy/helm`](deploy/helm) — Deployment, Service, HPA, and a Prometheus-Operator `ServiceMonitor`; DB URL from a secret; probes on `/healthz`.
- **CI**: GitHub Actions runs ruff + the full offline test suite on every push.

---

## Project layout

```
src/sovereign/
  config.py        typed settings (pydantic-settings)
  serving.py       LLM backends: echo / OpenAI-compatible (vLLM) / Bedrock
  embeddings.py    hashing / OpenAI embedders
  rag/             chunking · store (in-memory | pgvector) · ingest · retrieve · grounding
  agent.py         grounded answer flow + RagAgent tool surface
  mcp_server.py    MCP tools over the same primitives
  obs.py           Prometheus metrics + optional OpenTelemetry
  eval.py          offline eval harness (hit-rate / MRR / grounding)
  api.py           FastAPI application
deploy/            Dockerfile context, prometheus, grafana, helm
docs/adr/          architecture decision records
```

## Honest limitations

The offline defaults are **stand-ins, by design** — the hashing embedder is lexical (not
semantic), the reranker and grounding metric are lexical-overlap approximations, and the
in-memory store is O(n). They exist to make the *platform* runnable and testable anywhere; the
production path (semantic embeddings, a cross-encoder reranker / NLI grounding, pgvector HNSW)
swaps in behind the same interfaces. See [ADR-0003](docs/adr/0003-pluggable-backends.md) and
[ARCHITECTURE.md](ARCHITECTURE.md#scaling-path--limitations).

## License

MIT — see [LICENSE](LICENSE).