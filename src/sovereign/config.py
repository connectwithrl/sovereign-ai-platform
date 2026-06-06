"""Typed configuration for the platform.

Every backend choice is config-driven so the same image runs offline (deterministic
defaults, no external dependencies) or in production (vLLM/Bedrock + pgvector) by
changing environment variables only. See docs/adr/0003-pluggable-backends.md.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SOVEREIGN_", env_file=".env", extra="ignore")

    # --- Serving ---------------------------------------------------------------
    serving_backend: Literal["echo", "openai", "bedrock"] = "echo"
    # OpenAI-compatible endpoint — point at a local vLLM/TGI server in production.
    openai_base_url: str = "http://localhost:8001/v1"
    openai_api_key: str = "not-needed-for-local-vllm"
    serving_model: str = "Qwen/Qwen2.5-7B-Instruct"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    aws_region: str = "us-east-1"
    max_output_tokens: int = 512
    request_timeout_s: float = 60.0

    # --- Embeddings ------------------------------------------------------------
    embedding_backend: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256

    # --- Vector store ----------------------------------------------------------
    # When unset, an in-memory store is used (tests/demo). Set to a Postgres URL
    # to use pgvector in production, e.g. postgresql+psycopg://user:pw@host/db
    database_url: str | None = None
    vector_table: str = "documents"

    # --- Retrieval -------------------------------------------------------------
    top_k: int = 5
    rerank: bool = True
    rerank_candidates: int = 20
    chunk_tokens: int = 220
    chunk_overlap: int = 40

    # --- Observability ---------------------------------------------------------
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "sovereign-ai-platform"
    langfuse_enabled: bool = False

    # --- App -------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    # Default answer language hint for the bilingual (EN/AR) deployment context.
    default_language: Literal["en", "ar"] = "en"


@lru_cache
def get_settings() -> Settings:
    return Settings()