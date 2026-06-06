"""Observability: Prometheus metrics + optional OpenTelemetry tracing.

LLM platforms need more than RED metrics — the signals that matter are token throughput,
output-length and latency *distributions*, retrieval quality, and grounding/guardrail
outcomes. Those are first-class here so dashboards and SLOs can be built on them
(see deploy/grafana/dashboards). OTel tracing is opt-in via SOVEREIGN_OTEL_ENABLED.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
_TOKEN_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

REQUESTS = Counter(
    "sovereign_requests_total", "Requests by endpoint and outcome.", ["endpoint", "outcome"]
)
REQUEST_LATENCY = Histogram(
    "sovereign_request_latency_seconds", "End-to-end request latency.", ["endpoint"],
    buckets=_LATENCY_BUCKETS,
)
LLM_LATENCY = Histogram(
    "sovereign_llm_latency_seconds", "LLM generation latency.", ["model"], buckets=_LATENCY_BUCKETS
)
LLM_TOKENS = Counter(
    "sovereign_llm_tokens_total", "Tokens processed by kind.", ["model", "kind"]  # prompt|completion
)
OUTPUT_TOKENS = Histogram(
    "sovereign_llm_output_tokens", "Completion-token distribution.", ["model"], buckets=_TOKEN_BUCKETS
)
RETRIEVAL_LATENCY = Histogram(
    "sovereign_retrieval_latency_seconds", "Vector retrieval latency.", buckets=_LATENCY_BUCKETS
)
RETRIEVAL_TOPSCORE = Histogram(
    "sovereign_retrieval_top_score", "Top retrieval score per query.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
GROUNDING_SCORE = Histogram(
    "sovereign_grounding_score", "Grounding score per answer.",
    buckets=(0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0),
)
GROUNDING_BLOCKED = Counter(
    "sovereign_grounding_blocked_total", "Answers blocked by the grounding guardrail."
)


def record_completion(completion) -> None:
    LLM_TOKENS.labels(completion.model, "prompt").inc(completion.prompt_tokens)
    LLM_TOKENS.labels(completion.model, "completion").inc(completion.completion_tokens)
    OUTPUT_TOKENS.labels(completion.model).observe(completion.completion_tokens)


def setup_tracing(settings) -> None:
    """Best-effort OTel setup; never fails the app if exporters are unavailable."""
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
        )
        trace.set_tracer_provider(provider)
    except Exception:  # pragma: no cover - observability must never crash the app
        pass