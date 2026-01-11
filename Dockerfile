# ---------- BASE ----------
FROM python:3.13-slim AS mango-base

WORKDIR /app

# ---- Disable OpenTelemetry globally ----
ENV OTEL_SDK_DISABLED=true
ENV OTEL_TRACES_EXPORTER=none
ENV OTEL_METRICS_EXPORTER=none
ENV OTEL_LOGS_EXPORTER=none
ENV OTEL_EXPORTER_OTLP_ENDPOINT=

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV HF_HOME=/app/models

ARG EMBEDDING_MODEL
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}

# Pre-download SentenceTransformer model to cache it in the image
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='${EMBEDDING_MODEL}')"
RUN rm -rf /root/.cache/huggingface

# ---------- AGENT ----------
FROM mango-base AS mango-agent

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${SERVICE_PORT}"]

# ---------- MCP ----------
FROM mango-base AS mango-mcp
CMD ["python", "mcp_server/server.py"]
