"""Document ingestion: chunk -> embed -> upsert. Idempotent per doc_id."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sovereign.embeddings import Embedder
from sovereign.rag.chunking import chunk_text
from sovereign.rag.store import Record, VectorStore


@dataclass
class IngestResult:
    doc_id: str
    chunks: int
    replaced: int


def _doc_id(source: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return "doc-" + hashlib.blake2b(source.encode("utf-8"), digest_size=8).hexdigest()


def ingest_document(
    *,
    text: str,
    source: str,
    store: VectorStore,
    embedder: Embedder,
    doc_id: str | None = None,
    metadata: dict | None = None,
    chunk_tokens: int = 220,
    overlap: int = 40,
) -> IngestResult:
    did = _doc_id(source, doc_id)
    # Re-ingesting a document replaces its prior chunks (idempotent updates).
    replaced = store.delete_document(did)

    chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap=overlap)
    if not chunks:
        return IngestResult(doc_id=did, chunks=0, replaced=replaced)

    vectors = embedder.embed([c.text for c in chunks])
    records = [
        Record(
            id=f"{did}::{c.ordinal}",
            doc_id=did,
            text=c.text,
            ordinal=c.ordinal,
            metadata={"source": source, **(metadata or {})},
            embedding=vectors[i],
        )
        for i, c in enumerate(chunks)
    ]
    store.upsert(records)
    return IngestResult(doc_id=did, chunks=len(records), replaced=replaced)