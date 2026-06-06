"""FastAPI application — the platform's HTTP surface.

Endpoints:
    GET  /healthz            liveness/readiness
    GET  /metrics            Prometheus exposition
    POST /v1/ingest          ingest a document into the vector store
    POST /v1/search          vector retrieval (+ rerank)
    POST /v1/chat            grounded RAG answer with citations + grounding report
    DELETE /v1/documents/{id} remove a document's chunks

State (embedder, store, backend, agent) is built once from settings and attached to
app.state so it can be swapped in tests via dependency overrides.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from sovereign import obs
from sovereign.agent import RagAgent, answer_question
from sovereign.config import Settings, get_settings
from sovereign.embeddings import build_embedder
from sovereign.rag.ingest import ingest_document
from sovereign.serving import build_backend


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        obs.setup_tracing(settings)
        app.state.settings = settings
        app.state.embedder = build_embedder(settings)
        app.state.store = _build_store(settings)
        app.state.backend = build_backend(settings)
        app.state.agent = RagAgent(
            store=app.state.store,
            embedder=app.state.embedder,
            backend=app.state.backend,
            settings=settings,
        )
        yield

    app = FastAPI(title="Sovereign AI Platform", version="0.1.0", lifespan=lifespan)
    _register_routes(app)
    return app


def _build_store(settings: Settings):
    from sovereign.rag.store import build_store

    return build_store(settings)


# --- Schemas -----------------------------------------------------------------
class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str
    doc_id: str | None = None
    metadata: dict | None = None


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    replaced: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None


class SearchHit(BaseModel):
    text: str
    source: str
    doc_id: str
    score: float
    vector_score: float


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    language: str | None = None
    block_ungrounded: bool = False


class Citation(BaseModel):
    source: str
    doc_id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounding_score: float
    grounded: bool
    model: str
    usage: dict


# --- Routes ------------------------------------------------------------------
def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "documents": app.state.store.count()}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        with obs.REQUEST_LATENCY.labels("ingest").time():
            result = ingest_document(
                text=req.text,
                source=req.source,
                store=app.state.store,
                embedder=app.state.embedder,
                doc_id=req.doc_id,
                metadata=req.metadata,
                chunk_tokens=app.state.settings.chunk_tokens,
                overlap=app.state.settings.chunk_overlap,
            )
        obs.REQUESTS.labels("ingest", "ok").inc()
        return IngestResponse(doc_id=result.doc_id, chunks=result.chunks, replaced=result.replaced)

    @app.post("/v1/search", response_model=list[SearchHit])
    def search(req: SearchRequest) -> list[SearchHit]:
        start = time.perf_counter()
        hits = app.state.agent.search(req.query, top_k=req.top_k)
        obs.RETRIEVAL_LATENCY.observe(time.perf_counter() - start)
        if hits:
            obs.RETRIEVAL_TOPSCORE.observe(hits[0].score)
        obs.REQUESTS.labels("search", "ok").inc()
        return [
            SearchHit(
                text=h.text, source=h.source, doc_id=h.doc_id,
                score=round(h.score, 4), vector_score=round(h.vector_score, 4),
            )
            for h in hits
        ]

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if app.state.store.count() == 0:
            raise HTTPException(status_code=409, detail="knowledge base is empty; ingest documents first")
        with obs.REQUEST_LATENCY.labels("chat").time():
            llm_start = time.perf_counter()
            ans = answer_question(
                question=req.question,
                store=app.state.store,
                embedder=app.state.embedder,
                backend=app.state.backend,
                settings=app.state.settings,
                language=req.language,
                block_ungrounded=req.block_ungrounded,
            )
            obs.LLM_LATENCY.labels(ans.model).observe(time.perf_counter() - llm_start)
        obs.record_completion(
            type("C", (), {"model": ans.model, "prompt_tokens": ans.prompt_tokens,
                           "completion_tokens": ans.completion_tokens})
        )
        obs.GROUNDING_SCORE.observe(ans.grounding.score)
        if req.block_ungrounded and not ans.grounding.grounded:
            obs.GROUNDING_BLOCKED.inc()
        obs.REQUESTS.labels("chat", "ok").inc()
        return ChatResponse(
            answer=ans.text,
            citations=[Citation(source=c.source, doc_id=c.doc_id, score=round(c.score, 4))
                       for c in ans.citations],
            grounding_score=ans.grounding.score,
            grounded=ans.grounding.grounded,
            model=ans.model,
            usage={"prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens},
        )

    @app.delete("/v1/documents/{doc_id}")
    def delete_document(doc_id: str) -> dict:
        removed = app.state.store.delete_document(doc_id)
        obs.REQUESTS.labels("delete", "ok").inc()
        return {"doc_id": doc_id, "removed_chunks": removed}


app = build_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("sovereign.api:app", host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()