"""Token-aware document chunking.

Uses a lightweight word-count proxy for tokens (deterministic, dependency-free).
Splits on paragraph/sentence boundaries first, then packs spans up to `chunk_tokens`
with `overlap` carried between chunks to preserve cross-boundary context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    ordinal: int


def _segments(text: str, max_words: int) -> list[str]:
    """Split into sentence-ish segments, hard-splitting any segment longer than
    `max_words` so a single long sentence (or punctuation-free text) still chunks."""
    segs: list[str] = []
    for para in _PARA_RE.split(text.strip()):
        para = para.strip()
        if not para:
            continue
        for sent in (s.strip() for s in _SENT_RE.split(para) if s.strip()):
            words = sent.split()
            if len(words) <= max_words:
                segs.append(sent)
            else:
                for i in range(0, len(words), max_words):
                    segs.append(" ".join(words[i : i + max_words]))
    return segs


def chunk_text(text: str, chunk_tokens: int = 220, overlap: int = 40) -> list[Chunk]:
    if not text.strip():
        return []
    segments = _segments(text, max_words=chunk_tokens)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    ordinal = 0

    def flush() -> list[str]:
        nonlocal buf, buf_len, ordinal
        if not buf:
            return []
        chunks.append(Chunk(text=" ".join(buf), ordinal=ordinal))
        ordinal += 1
        # carry the tail words for overlap
        tail_words: list[str] = []
        count = 0
        for seg in reversed(buf):
            words = seg.split()
            tail_words[:0] = words
            count += len(words)
            if count >= overlap:
                break
        return [" ".join(tail_words)] if tail_words else []

    for seg in segments:
        seg_len = len(seg.split())
        if buf_len + seg_len > chunk_tokens and buf:
            carry = flush()
            buf = carry[:]
            buf_len = sum(len(c.split()) for c in buf)
        buf.append(seg)
        buf_len += seg_len
    flush()
    return chunks