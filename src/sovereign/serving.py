"""LLM serving backends.

The serving layer is an abstraction over inference providers. The default `EchoBackend`
is deterministic and offline so the platform runs in CI/demo without a model server;
`OpenAICompatBackend` targets a self-hosted vLLM/TGI server (the production default for
data-sovereign deployments); `BedrockBackend` targets AWS Bedrock.

`generate()` returns a `Completion` carrying token-usage so the observability layer can
record LLM-specific signals (token throughput, output length) — see obs.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _approx_tokens(text: str) -> int:
    # Cheap heuristic used only by the offline backend; real backends report usage.
    return max(1, len(text) // 4)


class ServingBackend(Protocol):
    model: str

    def generate(self, messages: list[Message], max_tokens: int) -> Completion: ...


class EchoBackend:
    """Deterministic offline backend.

    Produces a grounded, extractive answer from whatever context was supplied in the
    system/user prompt. This keeps the *pipeline* (retrieval -> prompt -> answer ->
    grounding eval) fully exercisable without a real model.
    """

    model = "echo-deterministic"

    def generate(self, messages: list[Message], max_tokens: int) -> Completion:
        question = next((m.content for m in reversed(messages) if m.role == "user"), "")
        context = "\n".join(m.content for m in messages if m.role == "system")
        # Extractive: return the context sentence with the most term overlap.
        answer = _extractive_answer(question, context) or "I don't have enough grounded context to answer."
        prompt_tokens = sum(_approx_tokens(m.content) for m in messages)
        return Completion(
            text=answer[: max_tokens * 4],
            prompt_tokens=prompt_tokens,
            completion_tokens=_approx_tokens(answer),
            model=self.model,
        )


def _extractive_answer(question: str, context: str) -> str:
    import re

    # Strip the prompt scaffolding (the "Context:" label and "[n] (source: ...)" markers)
    # so the extractive answer is a clean sentence from the source text, not the framing.
    context = re.sub(r"\bContext:\s*", " ", context)
    context = re.sub(r"\[\d+\]\s*\(source:[^)]*\)\s*", "", context)
    q_terms = set(re.findall(r"\w+", question.lower()))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    best, best_score = "", 0
    for s in sentences:
        score = len(q_terms & set(re.findall(r"\w+", s.lower())))
        if score > best_score:
            best, best_score = s, score
    return best


class OpenAICompatBackend:
    """Targets any OpenAI-compatible server — vLLM, TGI (OpenAI route), or OpenAI."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float) -> None:
        from openai import OpenAI  # lazy

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def generate(self, messages: list[Message], max_tokens: int) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            model=self.model,
        )


class BedrockBackend:
    """AWS Bedrock (Anthropic Claude) via the Converse API."""

    def __init__(self, model_id: str, region: str) -> None:
        import boto3  # lazy

        self.model = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def generate(self, messages: list[Message], max_tokens: int) -> Completion:
        system = [{"text": m.content} for m in messages if m.role == "system"]
        convo = [
            {"role": m.role, "content": [{"text": m.content}]}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        resp = self._client.converse(
            modelId=self.model,
            messages=convo,
            system=system,
            inferenceConfig={"maxTokens": max_tokens},
        )
        usage = resp.get("usage", {})
        text = resp["output"]["message"]["content"][0]["text"]
        return Completion(
            text=text,
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
            model=self.model,
        )


def build_backend(settings) -> ServingBackend:
    if settings.serving_backend == "openai":
        return OpenAICompatBackend(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.serving_model,
            timeout_s=settings.request_timeout_s,
        )
    if settings.serving_backend == "bedrock":
        return BedrockBackend(model_id=settings.bedrock_model_id, region=settings.aws_region)
    return EchoBackend()