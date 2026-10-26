# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
# Limits of Resolution in QML - reproducibility image (CPU only)
#
# Build:
#   docker build -t limits-qml .
#
# Cheap reproduction from the saved results (generates all paper figures into out/):
#   docker run --rm -v "$PWD/out:/app/out" limits-qml
#
# Fast 30-second pipeline audit (mounting checkpoints from host):
#   docker run --rm -v "$PWD/out:/app/out" -v "$PWD/checkpoints:/app/checkpoints" limits-qml python code/main.py --use-checkpoints --output out
#
# Full 82-claim numerical traceability audit:
#   docker run --rm limits-qml python scripts/build_traceability.py
#
# Any single statistical experiment:
#   docker run --rm limits-qml python code/local_experiments/exp10_paired_ci.py --help
# ──────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.5.0
# For bit-for-bit rebuilds also pin the base image by digest: python:3.12-slim@sha256:<digest>

# ── STAGE 1: dependencies ─────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy
WORKDIR /build

COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /bin/uv

# Compilers only if some dependency has no wheel for your platform; otherwise delete this block.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

# Install exact universal dependencies from cryptographically verified lockfile
COPY requirements.lock .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --require-hashes --index-strategy unsafe-best-match -r requirements.lock

# Fail the build, not the run, if the environment is broken (smoke test)
RUN python -c "import torch, pennylane, numpy, scipy, sklearn; print('torch', torch.__version__, 'pennylane', pennylane.__version__)"

# ── STAGE 2: runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="Limits of Resolution in QML - reproducibility image" \
      org.opencontainers.image.source="https://github.com/enriquerb32/limits-of-resolution-qml" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="1.0"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
    PATH="/opt/venv/bin:$PATH" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Single thread by default: BLAS thread count can change floating-point summation order.
# Override with -e OMP_NUM_THREADS=N for speed; note that results may then differ in the last digits.

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app/out /app/audit /app/paper/figs && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser code/ ./code/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser results/ ./results/
COPY --chown=appuser:appuser paper/ ./paper/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser Makefile pyproject.toml README.md LICENSE* CITATION.cff* ./

USER appuser
VOLUME ["/app/out"]

# Default: regenerate figures and tables from saved results (minutes, no training)
CMD ["python", "scripts/make_paper_figs.py", "--results", "results", "--out", "out"]
