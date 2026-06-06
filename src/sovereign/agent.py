"""Retrieval-augmented answering and a tool-using agent loop.

`answer_question` is the core RAG flow: retrieve -> build a grounded prompt -> generate ->
evaluate grounding -> (optionally) block ungrounded answers. It is framework-agnostic and
runs fully offline with the echo backend, which is what the tests and demo exercise.

`RagAgent` wraps the same primitives as callable *tools* (search, answer). In production
this is where Pydantic AI / an MCP client drives multi-step tool use; the offline loop
here keeps the contract identical so the wiring is testable without a live model.
"""

from __future__ import annotations

from dataclasses import dataclass

from sovereign.config import Settings
from sovereign.embeddings import Embedder
from sovereign.rag.grounding import GroundingReport, evaluate_grounding
from sovereign.rag.retrieve import Retrieved, retrieve
from sovereign.serving import Message, ServingBackend

_SYSTEM_PROMPT = (
    "You are a careful assistant for a government knowledge base. Answer ONLY from the "
    "provided context. If the context does not contain the answer, say you don't know. "
    "Answer in {language}."
)


@dataclass
class Answer:
    text: str
    citations: list[Retrieved]
    grounding: GroundingReport
    model: str
    prompt_tokens: int
    completion_tokens: int


def _build_messages(question: str, contexts: list[Retrieved], language: str) -> list[Message]:
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {c.source}) {c.text}" for i, c in enumerate(contexts)
    )
    return [
        Message(role="system", content=_SYSTEM_PROMPT.format(language=language)),
        Message(role="system", content=f"Context:\n{context_block}"),
        Message(role="user", content=question),
    ]


def answer_question(
    *,
    question: str,
    store,
    embedder: Embedder,
    backend: ServingBackend,
    settings: Settings,
    language: str | None = None,
    block_ungrounded: bool = False,
) -> Answer:
    lang = language or settings.default_language
    contexts = retrieve(
        query=question,
        store=store,
        embedder=embedder,
        top_k=settings.top_k,
        rerank=settings.rerank,
        rerank_candidates=settings.rerank_candidates,
    )
    messages = _build_messages(question, contexts, lang)
    completion = backend.generate(messages, max_tokens=settings.max_output_tokens)
    report = evaluate_grounding(completion.text, [c.text for c in contexts])

    text = completion.text
    if block_ungrounded and not report.grounded:
        text = "I can't answer that from the available sources with sufficient confidence."

    return Answer(
        text=text,
        citations=contexts,
        grounding=report,
        model=completion.model,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
    )


class RagAgent:
    """Exposes platform capabilities as named tools — the surface an MCP server re-exports."""

    def __init__(self, *, store, embedder: Embedder, backend: ServingBackend, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.backend = backend
        self.settings = settings

    def search(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        return retrieve(
            query=query,
            store=self.store,
            embedder=self.embedder,
            top_k=top_k or self.settings.top_k,
            rerank=self.settings.rerank,
            rerank_candidates=self.settings.rerank_candidates,
        )

    def answer(self, question: str, language: str | None = None) -> Answer:
        return answer_question(
            question=question,
            store=self.store,
            embedder=self.embedder,
            backend=self.backend,
            settings=self.settings,
            language=language,
        )