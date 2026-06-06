"""Embedding backends.

`HashingEmbedder` is a deterministic, dependency-free embedder (a hashed bag-of-
n-grams projected to a fixed dimension and L2-normalised). It is NOT a semantic model
— it exists so the platform, tests, and CI run anywhere with no GPU, no API key, and
fully reproducible vectors. In production set SOVEREIGN_EMBEDDING_BACKEND=openai
(or wire a local embedding server). See docs/adr/0003-pluggable-backends.md.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # shape (len(texts), dim), float32
        ...


class HashingEmbedder:
    """Deterministic hashing embedder over unigrams + bigrams."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for gram in grams:
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class OpenAIEmbedder:
    """Embedder backed by an OpenAI-compatible embeddings endpoint.

    Imported lazily so the core package installs without the `serving` extra.
    """

    def __init__(self, model: str, dim: int, base_url: str, api_key: str) -> None:
        from openai import OpenAI  # lazy

        self.dim = dim
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)


def build_embedder(settings) -> Embedder:
    if settings.embedding_backend == "openai":
        return OpenAIEmbedder(
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    return HashingEmbedder(dim=settings.embedding_dim)