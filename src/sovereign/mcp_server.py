"""Model Context Protocol (MCP) server exposing the platform as agent tools.

Re-exports the platform's retrieval/answer capabilities as MCP tools so any MCP client
(Claude Code, Claude Desktop, a Pydantic AI agent) can drive the knowledge base. This is
the same pattern as a production "Platform MCP Server": thin tool wrappers over the same
primitives the HTTP API uses, so there is one implementation and one set of guarantees.

Run:  sovereign-mcp        (stdio transport)
Requires the `agent` extra:  pip install -e ".[agent]"
"""

from __future__ import annotations

from sovereign.agent import RagAgent
from sovereign.config import get_settings
from sovereign.embeddings import build_embedder
from sovereign.rag.ingest import ingest_document
from sovereign.rag.store import build_store
from sovereign.serving import build_backend


def build_agent() -> RagAgent:
    settings = get_settings()
    store = build_store(settings)
    embedder = build_embedder(settings)
    backend = build_backend(settings)
    return RagAgent(store=store, embedder=embedder, backend=backend, settings=settings)


def main() -> None:
    from mcp.server.fastmcp import FastMCP  # lazy: requires the `agent` extra

    settings = get_settings()
    agent = build_agent()
    mcp = FastMCP("sovereign-ai-platform")

    @mcp.tool()
    def search_documents(query: str, top_k: int = 5) -> list[dict]:
        """Search the knowledge base and return the most relevant chunks with sources."""
        return [
            {"text": h.text, "source": h.source, "doc_id": h.doc_id, "score": round(h.score, 4)}
            for h in agent.search(query, top_k=top_k)
        ]

    @mcp.tool()
    def answer_question(question: str, language: str = "en") -> dict:
        """Answer a question grounded in the knowledge base, with citations and a grounding score."""
        ans = agent.answer(question, language=language)
        return {
            "answer": ans.text,
            "grounded": ans.grounding.grounded,
            "grounding_score": ans.grounding.score,
            "citations": [{"source": c.source, "doc_id": c.doc_id} for c in ans.citations],
        }

    @mcp.tool()
    def ingest_text(text: str, source: str) -> dict:
        """Ingest a document into the knowledge base."""
        result = ingest_document(
            text=text, source=source, store=agent.store, embedder=agent.embedder,
            chunk_tokens=settings.chunk_tokens, overlap=settings.chunk_overlap,
        )
        return {"doc_id": result.doc_id, "chunks": result.chunks}

    mcp.run()


if __name__ == "__main__":
    main()