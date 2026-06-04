# syntax=docker/dockerfile:1

# ---- builder: resolve and install dependencies into a venv ----
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Keep uv predictable inside the image: copy (not link) into the venv and
# byte-compile so first request latency is lower.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first, against the lockfile only, so this layer is
# cached as long as pyproject.toml / uv.lock are unchanged.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now bring in the source and install the project itself.
COPY src ./src
COPY README.md ./README.md
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime: slim image with just the venv and app payload ----
FROM python:3.11-slim-bookworm AS runtime

# curl is used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd --create-home --uid 10001 app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface

# Bring over the prebuilt virtualenv and the runtime payload.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "grounded_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
