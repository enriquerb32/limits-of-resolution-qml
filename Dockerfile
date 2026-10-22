# ──────────────────────────────────────────────────────────────────────────────
# QML_Antigravity — Reproducible container for Q1 journal submission
# ──────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t qml_antigravity .
#
# Run full pipeline:
#   docker run -v $(pwd)/datos:/app/datos -v $(pwd)/resultados:/app/resultados \
#       qml_antigravity python main.py --datos /app/datos --salida /app/resultados
#
# Run a single control experiment:
#   docker run -v $(pwd)/resultados:/app/resultados \
#       qml_antigravity python experimentos_control.py --experiment pr-auc \
#       --ckpt-dir /app/resultados/checkpoints
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

LABEL maintainer="Enrique Real <enriquerb32@gmail.com>"
LABEL description="Physics-Informed QNN for extreme precipitation forecasting"
LABEL version="1.0"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────────
# gcc/g++ needed by some pip packages (e.g., pmdarima, xgboost from source)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY code/ ./code/
COPY scripts/ ./scripts/
COPY results/ ./results/
COPY paper/ ./paper/
COPY make_figs.py .
COPY README.md .

# ── Default: generate figures and verify environment ─────────────────────────
CMD ["python", "make_figs.py"]
