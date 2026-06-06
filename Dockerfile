# syntax=docker/dockerfile:1

# Sovereign AI Platform — runtime image.
# Lean single-stage build. psycopg[binary] ships its own libpq, so no apt
# build dependencies are required for the pgvector retrieval path.
FROM python:3.12-slim

# Keep Python output unbuffered and skip writing .pyc files in the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy only what the build backend (hatchling) needs to produce the wheel.
# README.md is referenced by pyproject's `readme` field, so it must be present
# for the install to succeed.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package with the production pgvector retrieval extra.
RUN pip install --no-cache-dir ".[pgvector]"

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 sovereign \
    && chown -R sovereign:sovereign /app
USER sovereign

EXPOSE 8000

# Container-level health check against the app's liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/healthz', timeout=4).status == 200 else sys.exit(1)"

# sovereign-api launches uvicorn on SOVEREIGN_HOST:SOVEREIGN_PORT (default 0.0.0.0:8000).
CMD ["sovereign-api"]