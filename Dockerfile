FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv by copying pinned binary (best practice)
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

# Copy project metadata first for caching
COPY pyproject.toml README.md ./

# Sync dependencies into project venv for faster cold starts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Copy source and data
ADD . /app

# Final sync including project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]


