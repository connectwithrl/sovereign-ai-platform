"""Vector store abstraction with two implementations.

`InMemoryStore` (default) is used for tests and the offline demo. `PgVectorStore` is the
production path: PostgreSQL + pgvector with an HNSW index and cosine distance.
Both expose the same interface so the rest of the platform is storage-agnostic.

The pgvector choice (over a dedicated vector DB) is documented in
docs/adr/0002-pgvector-over-dedicated-vector-db.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Record:
    id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class ScoredRecord:
    record: Record
    score: float  # cosine similarity in [-1, 1]


class VectorStore(Protocol):
    def upsert(self, records: list[Record]) -> None: ...
    def query(self, embedding: np.ndarray, top_k: int) -> list[ScoredRecord]: ...
    def delete_document(self, doc_id: str) -> int: ...
    def count(self) -> int: ...


class InMemoryStore:
    """Exact cosine-similarity search over an in-memory matrix. O(n) per query."""

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def upsert(self, records: list[Record]) -> None:
        for r in records:
            if r.embedding is None:
                raise ValueError(f"record {r.id} has no embedding")
            self._records[r.id] = r

    def query(self, embedding: np.ndarray, top_k: int) -> list[ScoredRecord]:
        if not self._records:
            return []
        ids = list(self._records)
        matrix = np.vstack([self._records[i].embedding for i in ids])  # already normalised
        q = embedding / (np.linalg.norm(embedding) or 1.0)
        scores = matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [ScoredRecord(record=self._records[ids[i]], score=float(scores[i])) for i in order]

    def delete_document(self, doc_id: str) -> int:
        to_drop = [rid for rid, r in self._records.items() if r.doc_id == doc_id]
        for rid in to_drop:
            del self._records[rid]
        return len(to_drop)

    def count(self) -> int:
        return len(self._records)


class PgVectorStore:
    """PostgreSQL + pgvector store. SQLAlchemy/pgvector imported lazily."""

    def __init__(self, database_url: str, table: str, dim: int) -> None:
        from pgvector.sqlalchemy import Vector  # lazy
        from sqlalchemy import (
            Column,
            Integer,
            MetaData,
            String,
            Table,
            create_engine,
        )
        from sqlalchemy.dialects.postgresql import JSONB

        self._engine = create_engine(database_url, pool_pre_ping=True, pool_size=10)
        self._dim = dim
        metadata = MetaData()
        self._table = Table(
            table,
            metadata,
            Column("id", String, primary_key=True),
            Column("doc_id", String, index=True),
            Column("text", String),
            Column("ordinal", Integer),
            Column("metadata", JSONB),
            Column("embedding", Vector(dim)),
        )
        metadata.create_all(self._engine)
        self._ensure_index(table)

    def _ensure_index(self, table: str) -> None:
        from sqlalchemy import text

        # Cosine-distance HNSW index for approximate nearest-neighbour search.
        stmt = text(
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw "
            f"ON {table} USING hnsw (embedding vector_cosine_ops)"
        )
        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(stmt)

    def upsert(self, records: list[Record]) -> None:
        from sqlalchemy.dialects.postgresql import insert

        rows = [
            {
                "id": r.id,
                "doc_id": r.doc_id,
                "text": r.text,
                "ordinal": r.ordinal,
                "metadata": r.metadata,
                "embedding": r.embedding,
            }
            for r in records
        ]
        stmt = insert(self._table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={c: stmt.excluded[c] for c in ("text", "ordinal", "metadata", "embedding")},
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def query(self, embedding: np.ndarray, top_k: int) -> list[ScoredRecord]:
        from sqlalchemy import select

        dist = self._table.c.embedding.cosine_distance(embedding).label("dist")
        stmt = select(self._table, dist).order_by(dist).limit(top_k)
        out: list[ScoredRecord] = []
        with self._engine.connect() as conn:
            for row in conn.execute(stmt).mappings():
                out.append(
                    ScoredRecord(
                        record=Record(
                            id=row["id"],
                            doc_id=row["doc_id"],
                            text=row["text"],
                            ordinal=row["ordinal"],
                            metadata=row["metadata"] or {},
                        ),
                        score=1.0 - float(row["dist"]),  # cosine_distance -> similarity
                    )
                )
        return out

    def delete_document(self, doc_id: str) -> int:
        from sqlalchemy import delete

        with self._engine.begin() as conn:
            res = conn.execute(delete(self._table).where(self._table.c.doc_id == doc_id))
            return res.rowcount or 0

    def count(self) -> int:
        from sqlalchemy import func, select

        with self._engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(self._table)).scalar() or 0)


def build_store(settings) -> VectorStore:
    if settings.database_url:
        return PgVectorStore(
            database_url=settings.database_url,
            table=settings.vector_table,
            dim=settings.embedding_dim,
        )
    return InMemoryStore()