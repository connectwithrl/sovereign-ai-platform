"""Sovereign AI Platform — a self-hostable RAG + agent serving platform.

The package is organised by platform concern, each behind a small interface so the
offline/default implementation can be swapped for a production backend via config:

    config      - typed settings (pydantic-settings)
    serving     - LLM inference backends (echo / OpenAI-compatible (vLLM) / Bedrock)
    embeddings  - embedding backends (deterministic hashing / OpenAI)
    rag         - chunking, vector store, ingestion, retrieval, grounding eval
    agent       - retrieval-augmented agent loop over a serving backend
    mcp_server  - Model Context Protocol server exposing platform tools
    obs         - OpenTelemetry tracing + Prometheus metrics
    eval        - offline evaluation harness for the retrieval/answer pipeline
    api         - FastAPI application wiring the above together
"""

__version__ = "0.1.0"
