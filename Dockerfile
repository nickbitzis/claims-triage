# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv, app code, and config.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 claims
WORKDIR /app
COPY --from=build --chown=claims:claims /app /app
COPY --chown=claims:claims config ./config
# The batch job and the API write artifacts and the SQLite audit store under /app/data.
RUN mkdir -p /app/data && chown claims:claims /app/data
USER claims
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8834 CT_DATA_DIR=/app/data CT_CONFIG=/app/config/routing.yaml
EXPOSE 8834
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8834/healthz', timeout=2)"
CMD ["claimstriage", "serve"]
