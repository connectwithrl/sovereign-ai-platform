# 1. Record architecture decisions

## Status

Accepted

## Context

This platform makes a number of non-obvious engineering choices — running fully
offline by default, using PostgreSQL + pgvector instead of a dedicated vector
database, blending lexical and vector scores in retrieval, treating grounding as
a runtime guardrail. Future maintainers (and the author, months later) need to
know *why* each choice was made, not just *what* the code does, so that a change
that quietly violates an earlier decision can be recognised as such.

Long design documents rot and rarely get read. A lightweight, append-only log of
short records kept next to the code is enough.

## Decision

We record significant architecture decisions as Architecture Decision Records
(ADRs) using Michael Nygard's format. Each ADR is a numbered Markdown file under
`docs/adr/` with the structure:

- **Title** — a short noun phrase, prefixed with the ADR number.
- **Status** — `Proposed`, `Accepted`, `Superseded by NNNN`, or `Deprecated`.
- **Context** — the forces at play: the problem, constraints, and assumptions.
- **Decision** — the choice made, stated in the active voice ("We will...").
- **Consequences** — what becomes easier and what becomes harder, stated
  honestly, including accepted trade-offs.

ADRs are immutable once accepted. A decision is changed by writing a new ADR that
supersedes the old one, leaving the original in place as a historical record.

## Consequences

- The reasoning behind each major choice is discoverable in one place and
  versioned alongside the code that implements it.
- The format is cheap enough that there is no excuse to skip it for a real
  decision, and structured enough to be skimmable.
- The numbered, append-only convention means the history of how the design
  evolved is never lost to an edit.