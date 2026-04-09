FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8050

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + bundled dataset + trained model
COPY app.py .
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8050

# Healthcheck — Dash serves the layout at /
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8050'); sys.exit(0)" || exit 1

CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "--workers", "2", "--timeout", "60"]
