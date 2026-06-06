# 2. PostgreSQL + pgvector over a dedicated vector database

## Status

Accepted

## Context

The platform needs a production vector store for retrieval. The obvious
alternatives are a dedicated vector database (Qdrant, Pinecone, Weaviate, Milvus)
or a vector extension to a database the team already operates.

The deployment target is a data-sovereign, regulated environment. That reframes
the decision in ways that matter more than raw ANN throughput:

- **Sovereignty and compliance surface.** Every component inside the boundary has
  to be certified, patched, backed up, and audited. A dedicated vector service is
  another piece of infrastructure to operate and another data store that must be
  proven to keep regulated data inside the boundary. Fewer moving parts is a
  direct compliance and operational benefit, not just a convenience.
- **The team already runs PostgreSQL.** Backups, HA, access control, monitoring,
  and on-call runbooks already exist. pgvector reuses all of it.
- **Transactional consistency.** Documents and their chunk vectors can be written
  and deleted in the same database, in the same transaction. Ingestion is
  idempotent per `doc_id` (`rag/ingest.py` deletes prior chunks then upserts);
  keeping the source-of-truth metadata and the vectors in one store avoids the
  dual-write consistency problem of an external index.
- **Expected scale.** At the document volumes anticipated for a public-sector
  knowledge base, an HNSW index with cosine distance in pgvector is sufficient
  for the recall and latency we need.

## Decision

Use **PostgreSQL + pgvector** as the production vector store by default.

`PgVectorStore` (`src/sovereign/rag/store.py`) creates the table and an HNSW index
over the embedding column using `vector_cosine_ops`, and queries with
`cosine_distance` converted to a similarity score. It is selected automatically by
`build_store(settings)` whenever `SOVEREIGN_DATABASE_URL` is set; otherwise the
in-memory store is used for tests and the offline demo.

Crucially, both implementations sit behind the same `VectorStore` protocol
(`upsert` / `query` / `delete_document` / `count`). The rest of the platform is
storage-agnostic, so a dedicated ANN service can be introduced later behind the
same interface if scale demands it.

## Consequences

- The operational and compliance surface stays small: one database technology to
  certify, secure, back up, and operate.
- Documents and vectors are consistent by construction, with no external index to
  reconcile.
- Honest trade-off: a dedicated ANN service will outperform pgvector at very large
  scale. As the corpus grows into the tens of millions of vectors, HNSW build
  time, memory footprint, and the recall/latency trade-off under heavy
  concurrency become real constraints, and tuning (`m`, `ef_construction`,
  `ef_search`) only goes so far. There is a scale ceiling beyond which a purpose-
  built vector DB earns its added operational cost.
- The `VectorStore` interface keeps that option open. The decision is to default
  to the simpler thing and switch deliberately when measurements — not
  speculation — show the ceiling has been reached.